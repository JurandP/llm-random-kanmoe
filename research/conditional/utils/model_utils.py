from functools import partial
from typing import Literal, Optional
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lizrd.core import llm
from research.conditional.archive.continuous_moe_alternatives import (
    ContinuousMoEQuickMergeDifferentlySimple,
    ContinuousMoEQuickMergeDifferentlyCommonBase,
    ContinuousMoEQuickRawmerge,
    ContinuousMoEQuickTopmerge,
    ContinuousMoEQuickNosoftmax,
    ContinuousMoEQuickAdaTemp,
    ContinuousMoELayernorm,
    ContinuousMoEFinal,
    ContinuousMoERandomGroups,
)
from research.conditional.moe_layers.continuous_moe import (
    ContinuousMoE,
    ContinuousMoEQuick,
    FeedForwardTimed,
)
from research.conditional.moe_layers.expert_choice import ExpertChoiceFF


def make_loss_function(
    model: Literal["bert", "gpt"],
    loss_checkpoint_chungs: int,
    mask_percentage: Optional[float] = None,
):
    if model == "bert":
        assert mask_percentage is not None, "Mask percentage must be specified for BERT"
        if loss_checkpoint_chungs == 0:
            return partial(calculate_bert_loss, mask_percent=mask_percentage)
        else:
            return partial(
                chungized_bert_loss,
                mask_percent=mask_percentage,
                n_chungs=loss_checkpoint_chungs,
            )
    elif model == "gpt":
        if loss_checkpoint_chungs == 0:
            return calculate_gpt_loss
        else:
            return partial(chungized_gpt_loss, n_chungs=loss_checkpoint_chungs)
    else:
        raise ValueError(f"Model type {model} not implemented")


def chungized_llm_loss(
    input_tokens,
    gt_tokens,
    mask,
    model,
    mixed_precision,
    vocab_size,
    n_chungs,
    mask_percent,
):
    def make_custom_forward(vocab_size):
        def custom_forward(*inputs):
            with torch.autocast(
                device_type="cuda", enabled=mixed_precision, dtype=torch.float16
            ):
                output = model.head(inputs[0])
                gt = inputs[1]
                mask = inputs[2]
                loss = F.cross_entropy(
                    output.reshape(-1, vocab_size),
                    gt.reshape(-1).long(),
                    reduction="none",
                )
            return loss * mask.reshape(-1)

        return custom_forward

    encoder_output = model.encoder(input_tokens)

    chunged_inputs = torch.chunk(encoder_output, n_chungs, dim=0)
    chunged_non_masked_inputs = torch.chunk(gt_tokens, n_chungs, dim=0)
    chunged_non_masked_masks = torch.chunk(mask, n_chungs, dim=0)

    num_tokens = 0
    total_loss = 0
    for chunged_input, chunged_gt, chunged_mask in zip(
        chunged_inputs, chunged_non_masked_inputs, chunged_non_masked_masks
    ):
        partial_loss_output = checkpoint(
            make_custom_forward(vocab_size), chunged_input, chunged_gt, chunged_mask
        )

        num_tokens += partial_loss_output.shape[0]
        total_loss += partial_loss_output.sum()
    return total_loss / num_tokens / mask_percent


def chungized_bert_loss(
    batch, model, mixed_precision, vocab_size, mask_percent, n_chungs
):
    return chungized_llm_loss(
        input_tokens=batch.masked_tokens,
        gt_tokens=batch.tokens,
        mask=batch.mask_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
        n_chungs=n_chungs,
        mask_percent=mask_percent,
    )


def chungized_gpt_loss(batch, model, mixed_precision, vocab_size, n_chungs):
    return chungized_llm_loss(
        input_tokens=batch.tokens,
        gt_tokens=batch.target_tokens,
        mask=batch.non_padded_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
        n_chungs=n_chungs,
        mask_percent=1.0,
    )


def calculate_llm_loss(
    input_tokens, gt_tokens, mask, model, mixed_precision, vocab_size, mask_percent
):
    with torch.autocast(
        device_type="cuda", enabled=mixed_precision, dtype=torch.float16
    ):
        model_output = model(input_tokens)

    mask_loss = F.cross_entropy(
        model_output.reshape(-1, vocab_size),
        gt_tokens.reshape(-1).long(),
        reduction="none",
    )
    mask_loss *= mask.reshape(-1)
    loss = mask_loss.mean() / mask_percent
    return loss


def calculate_gpt_loss(batch, model, mixed_precision, vocab_size):
    return calculate_llm_loss(
        input_tokens=batch.tokens,
        gt_tokens=batch.target_tokens,
        mask=batch.non_padded_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
        mask_percent=1.0,
    )


def calculate_bert_loss(batch, model, mixed_precision, vocab_size, mask_percent):
    return calculate_llm_loss(
        input_tokens=batch.masked_tokens,
        gt_tokens=batch.tokens,
        mask=batch.mask_mask,
        model=model,
        mixed_precision=mixed_precision,
        vocab_size=vocab_size,
        mask_percent=mask_percent,
    )


def get_attention_layer(args):
    if args.model_type == "gpt":
        attention_layer_fun = lambda: llm.CausalAttention(args.dmodel, args.n_att_heads)
    elif args.model_type == "bert":
        attention_layer_fun = lambda: llm.Attention(args.dmodel, args.n_att_heads)
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")
    return attention_layer_fun


def get_expert_choice_args(args):
    if args.granularity_expert_config:
        if (args.expert_size is not None) or (args.topk_fraction is not None):
            raise ValueError(
                "Cannot specify expert_size or topk_fraction when using granularity config"
            )

        expert_size = args.total_experts_width / args.n_experts
        assert expert_size == int(expert_size)
        expert_size = int(expert_size)

        experts_per_token = args.effective_dff / expert_size

        topk_fraction = experts_per_token / args.n_experts
        assert 0.0 <= topk_fraction <= 1.0
    else:
        expert_size = args.expert_size
        topk_fraction = args.topk_fraction

    return {
        "dmodel": args.dmodel,
        "n_experts": args.n_experts,
        "expert_size": expert_size,
        "topk_fraction": topk_fraction,
        "random_perm": args.expert_random_perm,
    }


def get_ff_layer(args):
    if args.ff_mode == "vanilla":
        return_fn = lambda: llm.FeedForward(args.dmodel, args.dff)
    elif args.ff_mode == "vanilla_timed":
        return_fn = lambda: FeedForwardTimed(args.dmodel, args.dff)
    elif args.ff_mode == "cont_moe":
        return_fn = lambda: ContinuousMoE(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_quick":
        return_fn = lambda: ContinuousMoEQuick(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_quick_merge_diff_simple":
        return_fn = lambda: ContinuousMoEQuickMergeDifferentlySimple(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_quick_merge_diff_comm_base":
        return_fn = lambda: ContinuousMoEQuickMergeDifferentlyCommonBase(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_quick_rawmerge":
        return_fn = lambda: ContinuousMoEQuickRawmerge(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_quick_topmerge":
        return_fn = lambda: ContinuousMoEQuickTopmerge(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_quick_nosoft":
        return_fn = lambda: ContinuousMoEQuickNosoftmax(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_quick_adatemp":
        return_fn = lambda: ContinuousMoEQuickAdaTemp(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
            share_by_experts=args.share_by_experts,
            share_by_emit_merge=args.share_by_emit_merge,
        )
    elif args.ff_mode == "cont_moe_ln":
        return_fn = lambda: ContinuousMoELayernorm(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_final":
        return_fn = lambda: ContinuousMoEFinal(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
        )
    elif args.ff_mode == "cont_moe_random_groups":
        return_fn = lambda: ContinuousMoERandomGroups(
            dm=args.dmodel,
            dff=args.dff,
            n_experts=args.n_experts,
            group_size=args.group_size,
            sparsity_dim=args.sparsity_dim,
            temperature=args.temperature,
            expert_size=args.expert_size,
            use_opt_einsum=args.use_opt_einsum,
            flop_matched=args.flop_matched,
            batch_size=args.batch_size,
            seqlen=args.cutoff,
            mix_whole_batch=args.mix_whole_batch,
        )
    elif args.ff_mode == "expert_choice":
        return_fn = lambda: ExpertChoiceFF(
            **get_expert_choice_args(args),
        )
    else:
        raise NotImplementedError(f"FF mode {args.ff_mode} not implemented")

    if args.every_other_layer:
        if args.standard_ff_first:
            return_fn = llm.EveryOtherLayer(
                lambda: llm.FeedForward(args.dmodel, args.dff), return_fn
            )
        else:
            return_fn = llm.EveryOtherLayer(
                return_fn, lambda: llm.FeedForward(args.dmodel, args.dff)
            )

    return return_fn

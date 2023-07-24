import torch


def introduce_parser_arguments(parser):
    # core hyperparameters, fixed for all experiments; needs a good reason to change

    parser.add_argument("--use_clearml", action="store_true")
    parser.add_argument("--use_neptune", action="store_true")
    parser.add_argument("--batch_size", type=int, default=600)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cutoff", type=int, default=128)
    parser.add_argument("--dmodel", type=int, default=768)
    parser.add_argument("--dff", type=int, default=3072)
    parser.add_argument("--n_att_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=4)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--logging_interval_light", type=int, default=1000000)
    parser.add_argument("--logging_interval_heavy", type=int, default=1000000)
    parser.add_argument("--mask_loss_weight", type=float, default=1.0)
    parser.add_argument("--mask_percent", type=float, default=0.15)
    parser.add_argument("--n_steps", type=int, default=90000)
    parser.add_argument("--data_seed", type=int, default=42)
    parser.add_argument("--torch_seed", type=int, default=42)
    parser.add_argument("--tags", nargs="*", type=str, default=None)
    parser.add_argument(
        "--model_type", type=str, choices=["gpt", "bert"], default="bert"
    )

    # parameters usually changed for experiments

    parser.add_argument("--ff_mode", type=str, default="vanilla")
    parser.add_argument("--project_name", type=str, default="")
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--loss_checkpoint_chungs", type=str, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--n_experts", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=1)
    parser.add_argument("--sparsity_dim", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--expert_size", type=int, required=False)
    parser.add_argument("--topk_fraction", type=float, required=False)
    parser.add_argument("--logging_interval_loss", type=int, default=250)
    parser.add_argument("--every_other_layer", action="store_true")
    parser.add_argument("--expert_random_perm", action="store_true")
    parser.add_argument("--standard_ff_first", action="store_true")
    parser.add_argument("--granularity_expert_config", action="store_true")
    parser.add_argument("--total_experts_width", type=int, required=False)
    parser.add_argument("--effective_dff", type=int, required=False)
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--use_opt_einsum", action="store_true")
    parser.add_argument("--share_by_experts", action="store_true")
    parser.add_argument("--share_by_emit_merge", action="store_true")
    parser.add_argument("--flop_matched", action="store_true")
    parser.add_argument("--mix_whole_batch", action="store_true")

    # experimental/legacy parameters

    parser.add_argument("--hack_name", type=str, default=None)
    parser.add_argument("--x_flop", action="store_true")
    parser.add_argument("--x_logarithmic", action="store_true")

    return parser


def get_ith_chunk(tensor, chunks, i):
    list_of_chunks = torch.chunk(tensor, chunks, dim=0)
    return list_of_chunks[i]

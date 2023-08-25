import shutil
import os
import subprocess

from git import Repo, GitCommandError


def version_and_copy_code(
    newdir_name,
    name_for_branch,
    remote_name="cemetery",
    remote_url="git@github.com:Simontwice/llm-random-cemetery.git",
):
    """
    Stashes the current code, adds all changes, commits them, pushes them to a remote repo, and returns to the original branch.
    Then copies all code to a new directory.
    Prerequisite: the user needs to be able to push to the remote repo from the command line without entering a password.
    If not met, the user needs to set up ssh keys.
    """
    # Version code
    version_code(name_for_branch, remote_name, remote_url)

    # Copy code
    root_dir = find_git_root()
    newdir_path = f"{os.path.dirname(root_dir)}/sparsity_code_cemetery/{newdir_name}"

    ignore_patterns_file = os.path.join(root_dir, ".versioningignore")
    versioning_ignore_patterns = make_ignore_patterns(ignore_patterns_file)

    print(f"Copying code to {newdir_path}...")
    # Copy the project root directory to a new directory, ignoring files described in versioning_ignore_patterns
    shutil.copytree(root_dir, newdir_path, ignore=versioning_ignore_patterns)
    print(f"Code copied successfully to {newdir_path}")

    # Change to the new directory
    os.chdir(newdir_path)


def version_code(name_for_branch, remote_name, remote_url):
    repo = Repo(find_git_root())

    # Record current branch
    current_branch = repo.active_branch.name

    # Add remote if it does not exist
    check_remote(repo, remote_name, remote_url)

    # make new branch
    repo.git.checkout(b=name_for_branch)

    # update main
    repo.git.checkout("main")
    repo.git.pull()

    repo.git.checkout(name_for_branch)
    repo.git.merge("main")

    repo.git.push(remote_name, name_for_branch)

    repo.git.checkout(current_branch)


def rebase_on_new_main(name_for_branch, current_branch, repo):
    # Check for changes in the current workspace
    should_unstash = False
    if repo.is_dirty():
        should_unstash = True
        # Changes exist, so let's stash them
        try:
            repo.git.stash()
        except GitCommandError:
            raise GitCommandError(
                "Failed to stash changes. Not sure what to do. Aborting..."
            )

    # try to checkout main and pull
    try:
        # Switch to the 'main' branch
        repo.git.checkout("main")
    except GitCommandError:
        if should_unstash:
            repo.git.stash("pop")
        raise GitCommandError(
            "Failed to checkout main. Make sure you have a main branch."
        )
    try:
        # Perform git pull
        repo.git.pull()
    except GitCommandError:
        assert repo.active_branch.name == "main"
        repo.git.reset("--hard", "HEAD")
        repo.git.checkout(current_branch)
        if should_unstash:
            repo.git.stash("pop")
        raise GitCommandError(
            "Failed to pull from main. Make sure you have a main branch."
        )
    try:
        repo.git.checkout(b=name_for_branch)

    except GitCommandError:
        assert repo.active_branch.name == "main"
        repo.git.reset("--hard", "HEAD")
        repo.git.checkout(current_branch)
        if should_unstash:
            repo.git.stash("pop")
        raise GitCommandError("Failed to create new branch.")

    try:
        if should_unstash:
            try:
                # Use 'subprocess' module to call git stash pop
                subprocess.run(["git", "stash", "pop"], check=True)
            except subprocess.CalledProcessError as err:
                print(
                    "Error encountered while applying stashed changes.",
                    err,
                    "Trying to merge, favoring stash...",
                )
                try:
                    subprocess.run(["git", "checkout", "--theirs", "."], check=True)
                except subprocess.CalledProcessError as err:
                    print(
                        "Error encountered while resolving conflicts in favor of stash.",
                        err,
                    )
                    raise GitCommandError(
                        "Rebasing on new main failed, conflicts occurred. Could not resolve them automatically. "
                    )

    except GitCommandError:
        # In case of conflict or any other git error, reset back to the initial state
        repo.git.reset("--hard", "HEAD")
        # Reset back to the initial recorded state
        repo.git.checkout(current_branch)
        if should_unstash:
            # Unstash the changes
            repo.git.stash("pop")
        raise GitCommandError(
            "An error occurred. There are conflicts between newest main and your local changes. Resolve them locally (e.g. by merging the newest main to your current branch) and try again. \nResetting back to initial state..."
        )


def check_remote(repo, remote_name, remote_url):
    for remote in repo.remotes:
        if remote.name == remote_name:
            if remote.url == remote_url:
                return
            else:
                raise Exception(
                    f"Wrong url under remote repo {remote_name}: {repo.remotes[remote_name].url.strip()}, should be {remote_url}"
                )
    repo.create_remote(remote_name, url=remote_url)
    return


def find_git_root():
    current_dir = os.getcwd()
    while True:
        git_dir = os.path.join(current_dir, ".git")
        if os.path.exists(git_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            raise Exception("Not in a Git repository.")
        current_dir = parent_dir


def make_ignore_patterns(filepath):
    # Set up ignore patterns
    with open(filepath) as f:
        patterns = f.read().splitlines()
        patterns = [
            pattern for pattern in patterns if pattern != "" and pattern[0] != "#"
        ]
        patterns = [p.strip() for p in patterns]
        patterns = shutil.ignore_patterns(*patterns)
    return patterns

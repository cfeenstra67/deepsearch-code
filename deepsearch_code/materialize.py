import asyncio
import os
import tarfile
import tempfile
from pathlib import Path

from . import github_client, settings


def _should_extract_member(member: tarfile.TarInfo, repo_root_in_tar: str) -> bool:
    """Check if a tar member should be extracted (is a file, not ignored)."""
    if not member.isfile():
        return False  # Skip directories, symlinks, etc.

    # Path relative to the *repo root* inside the tarball
    relative_path = Path(member.name).relative_to(repo_root_in_tar).as_posix()

    # Check ignored directories
    parts = relative_path.split("/")
    if any(part in settings.CODE_IGNORE_DIRS for part in parts):
        # print(f"Ignoring dir: {relative_path}")
        return False

    # Check ignored extensions
    _, ext = os.path.splitext(relative_path)
    if ext.lower() in settings.CODE_IGNORE_EXTENSIONS:
        # print(f"Ignoring ext: {relative_path}")
        return False

    # Check file size (already checked in download, but double-check here)
    if member.size > settings.MAX_INDEXING_FILE_SIZE_BYTES:
        print(f"Skipping large file in tar: {relative_path} ({member.size} bytes)")
        return False

    return True


async def materialize_code_from_tarball(repo_name: str, target_base_dir: str) -> str:
    """
    Downloads and extracts repo code tarball to a target directory.
    Returns the path to the extracted code directory.
    """
    repo_dir_name = repo_name.replace("/", "_")  # Sanitize for directory name
    output_code_dir = os.path.join(target_base_dir, repo_dir_name, "code")
    os.makedirs(output_code_dir, exist_ok=True)  # Ensure target dir exists

    try:
        default_branch = await github_client.get_repo_default_branch(repo_name)
        print(f"Default branch for {repo_name}: {default_branch}")

        # Download tarball to a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".tar.gz"
        ) as tmp_tar_file:
            tmp_tar_path = tmp_tar_file.name

        await github_client.download_repo_tarball(
            repo_name, default_branch, tmp_tar_path
        )

        # Extract the tarball (synchronous for now, wrap in executor if becomes bottleneck)
        print(f"Extracting tarball {tmp_tar_path} to {output_code_dir}...")
        extracted_count = 0
        repo_root_in_tar = None
        try:
            # Use asyncio.to_thread for the blocking tarfile operation
            def extract_sync():
                nonlocal extracted_count, repo_root_in_tar
                count = 0
                root_found = False
                with tarfile.open(tmp_tar_path, "r:gz") as tar:
                    # Find the common root directory within the tarball
                    # GitHub tarballs usually have a single top-level folder like 'owner-repo-sha'
                    all_members = tar.getmembers()
                    if not all_members:
                        raise ValueError("Tarball is empty")

                    # Assume first member's first component is the root
                    repo_root_in_tar = Path(all_members[0].name).parts[0]
                    print(f"Detected tarball root directory: {repo_root_in_tar}")
                    root_found = True

                    for member in all_members:
                        if not root_found:  # Should not happen if tar isn't empty
                            raise ValueError(
                                "Could not determine tarball root directory"
                            )

                        if _should_extract_member(member, repo_root_in_tar):
                            # Calculate target path *relative* to output_code_dir
                            relative_path = Path(member.name).relative_to(
                                repo_root_in_tar
                            )
                            target_path = Path(output_code_dir) / relative_path

                            # Ensure parent directory exists before extracting
                            target_path.parent.mkdir(parents=True, exist_ok=True)

                            # Extract file content
                            try:
                                with (
                                    tar.extractfile(member) as source,
                                    open(target_path, "wb") as dest,
                                ):
                                    dest.write(source.read())
                                count += 1
                            except Exception as extract_err:
                                print(
                                    f"Error extracting file {member.name} to {target_path}: {extract_err}"
                                )
                                # Decide whether to continue or raise
                        # else:
                        #     print(f"Skipping tar member: {member.name}")
                return count

            extracted_count = await asyncio.to_thread(extract_sync)
            print(f"Extraction complete. Extracted {extracted_count} files.")

        except (tarfile.TarError, ValueError, OSError, Exception) as e:
            print(f"Error during tarball extraction: {e}")
            raise  # Re-raise error
        finally:
            # Clean up the temporary tarball file
            if os.path.exists(tmp_tar_path):
                os.remove(tmp_tar_path)

        return output_code_dir  # Return path to the extracted code

    except Exception as e:
        # Catch errors from get_repo_default_branch or download_repo_tarball
        print(f"Failed to materialize code for {repo_name}: {e}")
        raise  # Re-raise


# Add materialization for issues, PRs etc. later if needed
# async def materialize_issues(...)

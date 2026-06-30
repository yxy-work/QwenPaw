'''
压缩包上传后的内部解压 helper。

该模块面向 QwenPaw 上传链路使用，入口是一个已经落盘的压缩包文件，
输出目录默认取压缩包同路径去后缀后的目录。例如：

    /workspace/media/demo.zip -> /workspace/media/demo/

如默认输出目录已存在，会自动改用 demo__2、demo__3 等目录，避免覆盖
用户已有结果。解压完成后会写入 unpack_meta.json，记录源压缩包、实际
输出目录、递归解压结构、不安全成员跳过情况与去重结果。
'''

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import BadZipFile, ZipFile


SUPPORTED_SUFFIXES = {'.zip', '.rar'}
META_VERSION = 1
MAX_SAFE_COMPONENT_BYTES = 120
ZIP_UTF8_FLAG = 0x800
DEDUP_TEMP_SUFFIX = '.qwenpaw_unpack_dedup_tmp'


@dataclass(frozen=True)
class ArchiveTask:
    archive_path: Path
    logical_archive_path: PurePosixPath
    parent_archive_output: str | None
    output_dir: Path


@dataclass(frozen=True)
class ExtractedMember:
    member_path: PurePosixPath
    output_path: Path
    is_dir: bool


def is_supported_archive_path(path: str | Path) -> bool:
    archive_path = Path(path)
    return (
        archive_path.is_file()
        and archive_path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def make_posix_relative(path: Path, root: Path) -> PurePosixPath:
    return PurePosixPath(path.relative_to(root).as_posix())


def sanitize_dir_name(name: str) -> str:
    sanitized_name = re.sub(r'[\\/:\0]+', '_', name).strip()
    return sanitized_name or 'archive'


def truncate_utf8_bytes(value: str, max_bytes: int) -> str:
    encoded_value = value.encode('utf-8')
    if len(encoded_value) <= max_bytes:
        return value
    truncated_value = encoded_value[:max_bytes]
    return truncated_value.decode('utf-8', errors='ignore').rstrip()


def shorten_component(
    component: str,
    max_bytes: int = MAX_SAFE_COMPONENT_BYTES,
) -> str:
    sanitized_component = re.sub(r'[\0/\\]+', '_', component).strip()
    if not sanitized_component:
        sanitized_component = 'item'
    if len(sanitized_component.encode('utf-8')) <= max_bytes:
        return sanitized_component

    digest = hashlib.blake2s(
        sanitized_component.encode('utf-8'),
        digest_size=6,
    ).hexdigest()
    suffix = Path(sanitized_component).suffix
    if len(suffix.encode('utf-8')) > 24:
        suffix = ''
    stem = sanitized_component[: -len(suffix)] if suffix else sanitized_component
    reserved_bytes = len(digest.encode('utf-8')) + 2 + len(suffix.encode('utf-8'))
    prefix_bytes = max(8, max_bytes - reserved_bytes)
    prefix = truncate_utf8_bytes(stem, prefix_bytes).rstrip(' ._')
    if not prefix:
        prefix = 'item'
    return f'{prefix}__{digest}{suffix}'


def reserve_top_level_output_dir(archive_path: Path) -> tuple[Path, dict[str, Any]]:
    parent_dir = archive_path.parent
    base_name = shorten_component(sanitize_dir_name(archive_path.stem))
    requested_output_dir = parent_dir / base_name
    output_dir = requested_output_dir
    suffix_index = 2
    while output_dir.exists():
        output_dir = parent_dir / shorten_component(f'{base_name}__{suffix_index}')
        suffix_index += 1

    return output_dir, {
        'requested_output_root': str(requested_output_dir),
        'actual_output_root': str(output_dir),
        'renamed': output_dir != requested_output_dir,
    }


def reserve_nested_output_dir(
    root: Path,
    archive_name: str,
    used_output_names: set[str],
) -> Path:
    base_name = shorten_component(sanitize_dir_name(archive_name))
    output_name = base_name
    suffix_index = 2
    while output_name in used_output_names or (root / output_name).exists():
        output_name = shorten_component(f'{base_name}__{suffix_index}')
        suffix_index += 1
    used_output_names.add(output_name)
    return root / output_name


def validate_member_path(member_name: str) -> PurePosixPath | None:
    member_path = PurePosixPath(member_name)
    if member_path.is_absolute() or '..' in member_path.parts:
        return None
    if not member_path.parts:
        return None
    return member_path


def recover_legacy_zip_member_name(member_name: str) -> str:
    try:
        raw_name = member_name.encode('cp437')
    except UnicodeEncodeError:
        return member_name

    if raw_name.isascii():
        return member_name

    for encoding_name in ('gb18030', 'gbk', 'big5'):
        try:
            decoded_name = raw_name.decode(encoding_name)
        except UnicodeDecodeError:
            continue
        if decoded_name != member_name:
            return decoded_name
    return member_name


def decode_zip_member_name(archive_filename: str, flag_bits: int) -> str:
    if flag_bits & ZIP_UTF8_FLAG:
        return archive_filename
    return recover_legacy_zip_member_name(archive_filename)


def build_safe_member_path(member_path: PurePosixPath) -> Path:
    safe_parts = [shorten_component(part) for part in member_path.parts]
    return Path(*safe_parts)


def extract_zip(
    archive_path: Path,
    output_dir: Path,
) -> tuple[list[ExtractedMember], list[str]]:
    extracted_members = []
    skipped_members = []
    try:
        with ZipFile(archive_path) as archive_file:
            for archive_info in archive_file.infolist():
                member_name = decode_zip_member_name(
                    archive_info.filename,
                    archive_info.flag_bits,
                )
                member_path = validate_member_path(member_name)
                if member_path is None:
                    skipped_members.append(archive_info.filename)
                    continue
                target_path = output_dir / build_safe_member_path(member_path)
                if archive_info.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with archive_file.open(archive_info) as source_file:
                        with target_path.open('wb') as target_file:
                            shutil.copyfileobj(source_file, target_file)
                extracted_members.append(
                    ExtractedMember(
                        member_path=member_path,
                        output_path=target_path,
                        is_dir=archive_info.is_dir(),
                    )
                )
    except BadZipFile as exc:
        raise RuntimeError(f'ZIP 文件损坏或格式不受支持: {archive_path}') from exc
    return extracted_members, skipped_members


def extract_rar_with_external_tool(
    archive_path: Path,
    output_dir: Path,
) -> tuple[list[ExtractedMember], list[str]]:
    bsdtar_path = shutil.which('bsdtar')
    unrar_path = shutil.which('unrar')
    if bsdtar_path is not None:
        command = [bsdtar_path, '-xf', str(archive_path), '-C', str(output_dir)]
    elif unrar_path is not None:
        command = [unrar_path, 'x', '-y', str(archive_path), str(output_dir)]
    else:
        raise RuntimeError(
            '当前环境缺少 RAR 解压工具。请安装 bsdtar 或 unrar 后重试。'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            'RAR 解压失败: '
            f'{archive_path}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}'
        )

    extracted_members = []
    for output_path in sorted(output_dir.rglob('*')):
        relative_member = make_posix_relative(output_path, output_dir)
        extracted_members.append(
            ExtractedMember(
                member_path=relative_member,
                output_path=output_path,
                is_dir=output_path.is_dir(),
            )
        )
    return extracted_members, []


def extract_archive(
    archive_path: Path,
    output_dir: Path,
) -> tuple[list[ExtractedMember], list[str]]:
    suffix = archive_path.suffix.lower()
    if suffix == '.zip':
        return extract_zip(archive_path, output_dir)
    if suffix == '.rar':
        return extract_rar_with_external_tool(archive_path, output_dir)
    raise RuntimeError(f'不支持的压缩格式: {archive_path}')


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def should_deduplicate_file(path: Path, meta_name: str) -> bool:
    if not path.is_file():
        return False
    if path.name == meta_name:
        return False
    if path.name.endswith(DEDUP_TEMP_SUFFIX):
        return False
    if path.suffix.lower() in SUPPORTED_SUFFIXES:
        return False
    return True


def replace_with_hardlink(source_path: Path, duplicate_path: Path) -> bool:
    if source_path.samefile(duplicate_path):
        return False

    temp_path = duplicate_path.with_name(
        f'.{duplicate_path.name}.{source_path.stat().st_ino}{DEDUP_TEMP_SUFFIX}'
    )
    suffix_index = 2
    while temp_path.exists():
        temp_path = duplicate_path.with_name(
            f'.{duplicate_path.name}.{source_path.stat().st_ino}.'
            f'{suffix_index}{DEDUP_TEMP_SUFFIX}'
        )
        suffix_index += 1

    os.link(source_path, temp_path)
    duplicate_path.unlink()
    temp_path.rename(duplicate_path)
    return True


def deduplicate_output_files(root: Path, meta_name: str) -> dict[str, Any]:
    files_by_size: dict[int, list[Path]] = {}
    scanned_file_count = 0
    scanned_size = 0
    for path in sorted(root.rglob('*')):
        if not should_deduplicate_file(path, meta_name):
            continue
        file_size = path.stat().st_size
        scanned_file_count += 1
        scanned_size += file_size
        files_by_size.setdefault(file_size, []).append(path)

    files_by_hash: dict[str, list[Path]] = {}
    for same_size_files in files_by_size.values():
        if len(same_size_files) < 2:
            continue
        for path in same_size_files:
            files_by_hash.setdefault(file_sha256(path), []).append(path)

    duplicate_groups = [
        paths for paths in files_by_hash.values() if len(paths) > 1
    ]
    linked_duplicate_count = 0
    duplicate_extra_size = 0
    groups = []
    for paths in duplicate_groups:
        canonical_path = paths[0]
        duplicate_paths = paths[1:]
        group_linked_count = 0
        group_extra_size = 0
        for duplicate_path in duplicate_paths:
            if replace_with_hardlink(canonical_path, duplicate_path):
                linked_duplicate_count += 1
                group_linked_count += 1
                group_extra_size += canonical_path.stat().st_size
        duplicate_extra_size += group_extra_size
        groups.append(
            {
                'canonical_path': make_posix_relative(
                    canonical_path,
                    root,
                ).as_posix(),
                'duplicate_paths': [
                    make_posix_relative(path, root).as_posix()
                    for path in duplicate_paths
                ],
                'file_size': canonical_path.stat().st_size,
                'linked_duplicate_count': group_linked_count,
            }
        )

    return {
        'enabled': True,
        'method': 'hardlink',
        'scanned_file_count': scanned_file_count,
        'scanned_size': scanned_size,
        'duplicate_group_count': len(duplicate_groups),
        'duplicate_file_count': sum(len(paths) for paths in duplicate_groups),
        'duplicate_extra_count': sum(len(paths) - 1 for paths in duplicate_groups),
        'linked_duplicate_count': linked_duplicate_count,
        'duplicate_extra_size': duplicate_extra_size,
        'groups': groups,
    }


def find_nested_archives(
    extracted_members: Iterable[ExtractedMember],
    logical_root: PurePosixPath,
    parent_archive_output: str,
    output_root: Path,
    used_output_names: set[str],
) -> list[ArchiveTask]:
    nested_tasks = []
    for member in extracted_members:
        if member.is_dir or not is_supported_archive_path(member.output_path):
            continue
        nested_output_dir = reserve_nested_output_dir(
            output_root,
            member.output_path.stem,
            used_output_names,
        )
        nested_tasks.append(
            ArchiveTask(
                archive_path=member.output_path,
                logical_archive_path=logical_root / member.member_path,
                parent_archive_output=parent_archive_output,
                output_dir=nested_output_dir,
            )
        )
    return nested_tasks


def build_member_records(
    extracted_members: Iterable[ExtractedMember],
    root: Path,
    current_logical_root: PurePosixPath,
) -> list[dict[str, Any]]:
    records = []
    for member in extracted_members:
        records.append(
            {
                'archive_member_path': member.member_path.as_posix(),
                'actual_output_path': make_posix_relative(
                    member.output_path,
                    root,
                ).as_posix(),
                'logical_layered_path': (
                    current_logical_root / member.member_path
                ).as_posix(),
                'is_dir': member.is_dir,
            }
        )
    return records


def unpack_archive_file(
    archive_path: str | Path,
    meta_name: str = 'unpack_meta.json',
    deduplicate: bool = True,
) -> dict[str, Any]:
    source_archive = Path(archive_path).expanduser().resolve()
    if not source_archive.exists() or not source_archive.is_file():
        raise FileNotFoundError(f'目标路径不是有效文件: {source_archive}')
    if source_archive.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise RuntimeError(f'不支持的压缩格式: {source_archive}')

    output_root, conflict = reserve_top_level_output_dir(source_archive)
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    initial_logical_path = PurePosixPath(source_archive.name)
    queue = [
        ArchiveTask(
            archive_path=source_archive,
            logical_archive_path=initial_logical_path,
            parent_archive_output=None,
            output_dir=output_root,
        )
    ]
    used_output_names = {output_root.name}
    processed_physical_paths = set()
    metadata_entries = []

    while queue:
        task = queue.pop(0)
        resolved_archive_path = task.archive_path.resolve()
        if resolved_archive_path in processed_physical_paths:
            continue
        processed_physical_paths.add(resolved_archive_path)

        task.output_dir.mkdir(parents=True, exist_ok=True)
        current_logical_root = task.logical_archive_path.with_suffix('')
        extracted_members, skipped_members = extract_archive(
            task.archive_path,
            task.output_dir,
        )
        member_records = build_member_records(
            extracted_members=extracted_members,
            root=output_root,
            current_logical_root=current_logical_root,
        )
        output_relative_path = (
            '.'
            if task.output_dir == output_root
            else make_posix_relative(task.output_dir, output_root).as_posix()
        )
        metadata_entries.append(
            {
                'source_archive_actual_path': str(task.archive_path),
                'source_archive_logical_path': task.logical_archive_path.as_posix(),
                'parent_archive_output': task.parent_archive_output,
                'output_dir': output_relative_path,
                'logical_extract_root': current_logical_root.as_posix(),
                'member_count': len(member_records),
                'members': member_records,
                'skipped_unsafe_members': skipped_members,
            }
        )

        nested_tasks = find_nested_archives(
            extracted_members=extracted_members,
            logical_root=current_logical_root,
            parent_archive_output=output_relative_path,
            output_root=output_root,
            used_output_names=used_output_names,
        )
        queue.extend(nested_tasks)

    if deduplicate:
        deduplication = deduplicate_output_files(output_root, meta_name)
    else:
        deduplication = {'enabled': False}

    metadata = {
        'meta_version': META_VERSION,
        'input_archive': str(source_archive),
        'output_root': str(output_root),
        'archive_count': len(metadata_entries),
        'conflict': conflict,
        'deduplication': deduplication,
        'entries': metadata_entries,
    }
    meta_path = output_root / meta_name
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )

    return {
        'input_archive': str(source_archive),
        'output_root': str(output_root),
        'mode': 'apply',
        'archive_count': len(metadata_entries),
        'meta_path': str(meta_path),
        'entries': metadata_entries,
        'conflict': conflict,
        'deduplication': deduplication,
        'meta_written': True,
    }

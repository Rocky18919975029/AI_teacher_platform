"""Credential provisioning is private, distinct, and does not overwrite existing files."""

import json
import os
import stat

import pytest

from scripts import configure
from scripts.load_test import load_students
from teacher.config import Settings


def read_config(path):
    return Settings(_env_file=path)


def test_default_still_creates_admin_student_and_four_independent_secrets(tmp_path, capsys):
    path = tmp_path / '.env'
    assert configure.main(['--output', str(path)]) == 0
    cfg = read_config(path)
    assert len(cfg.api_keys) == 2
    assert [p.role for p in cfg.api_keys] == ['admin', 'student']
    assert [p.subject for p in cfg.api_keys] == ['instructor', 'student-001']
    assert all(p.courses == ['metaverse'] for p in cfg.api_keys)
    credentials = [p.key.get_secret_value() for p in cfg.api_keys]
    credentials.extend([cfg.vllm_api_key.get_secret_value(), cfg.qdrant_api_key])
    assert len(set(credentials)) == 4
    assert all(len(value) >= 32 and 'CHANGE_ME' not in value for value in credentials)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    captured = capsys.readouterr()
    assert all(value not in captured.out + captured.err for value in credentials)


def test_40_students_inventory_matches_scoped_distinct_principals(tmp_path, capsys):
    path, inventory = tmp_path / '.env', tmp_path / 'users.json'
    assert (
        configure.main(
            [
                '--output',
                str(path),
                '--students',
                '40',
                '--users-output',
                str(inventory),
                '--course-id',
                'physics',
            ]
        )
        == 0
    )
    cfg = read_config(path)
    assert len(cfg.api_keys) == 41
    assert len({p.subject for p in cfg.api_keys}) == 41
    assert len({p.key.get_secret_value() for p in cfg.api_keys}) == 41
    students, identity = load_students(inventory, 40, 'default')
    assert identity.startswith('distinct_keys')
    assert [student.key for student in students] == [p.key.get_secret_value() for p in cfg.api_keys[1:]]
    assert all(student.course_id == 'physics' for student in students)
    assert all(p.courses == ['physics'] for p in cfg.api_keys)
    assert all(p.role == 'student' for p in cfg.api_keys[1:])
    assert stat.S_IMODE(inventory.stat().st_mode) == stat.S_IMODE(path.stat().st_mode) == 0o600
    raw = json.loads(inventory.read_text())
    assert all(set(row) == {'key', 'course_id'} for row in raw)
    captured = capsys.readouterr()
    assert all(student.key not in captured.out + captured.err for student in students)


@pytest.mark.parametrize('existing', ['env', 'inventory'])
def test_existing_artifact_preserved_and_no_partner_left_behind(tmp_path, existing):
    path, inventory = tmp_path / '.env', tmp_path / 'users.json'
    protected = path if existing == 'env' else inventory
    absent = inventory if existing == 'env' else path
    protected.write_bytes(b'original data')
    before = protected.stat()
    assert configure.main(['--output', str(path), '--users-output', str(inventory)]) == 1
    assert protected.read_bytes() == b'original data'
    assert protected.stat().st_ino == before.st_ino
    assert not absent.exists()


def test_failure_during_second_write_rolls_back_both_new_files(tmp_path, monkeypatch):
    path, inventory = tmp_path / '.env', tmp_path / 'users.json'
    real_fsync = os.fsync
    calls = 0

    def fail_second(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('Simulated disk failure')
        real_fsync(descriptor)

    monkeypatch.setattr(configure.os, 'fsync', fail_second)
    assert configure.main(['--output', str(path), '--users-output', str(inventory)]) == 1
    assert calls == 2
    assert not path.exists() and not inventory.exists()


def test_second_output_missing_directory_rolls_back_first(tmp_path):
    path, inventory = tmp_path / '.env', tmp_path / 'missing' / 'users.json'
    assert configure.main(['--output', str(path), '--users-output', str(inventory)]) == 1
    assert not path.exists()


def test_rollback_does_not_delete_replacement_from_another_process(tmp_path, monkeypatch):
    path, inventory = tmp_path / '.env', tmp_path / 'users.json'
    real_fsync = os.fsync
    calls = 0

    def replace_and_fail(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            path.unlink()
            path.write_text('separate owner')
            raise OSError('Simulated competing writer')
        real_fsync(descriptor)

    monkeypatch.setattr(configure.os, 'fsync', replace_and_fail)
    assert configure.main(['--output', str(path), '--users-output', str(inventory)]) == 1
    assert path.read_text() == 'separate owner'
    assert not inventory.exists()


@pytest.mark.parametrize(
    'arguments', [['--students', '0'], ['--students', '257'], ['--course-id', "physics'\nINJECT=value"]]
)
def test_invalid_arguments_create_nothing(tmp_path, arguments):
    path = tmp_path / '.env'
    with pytest.raises(SystemExit) as error:
        configure.main(['--output', str(path), *arguments])
    assert error.value.code == 2
    assert not path.exists()


def test_output_paths_must_differ(tmp_path):
    path = tmp_path / '.env'
    with pytest.raises(SystemExit) as error:
        configure.main(['--output', str(path), '--users-output', str(path)])
    assert error.value.code == 2
    assert not path.exists()

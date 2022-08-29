import inspect
import json
import os
import sys
from abc import ABC
from threading import Lock
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar

import attr

from common.log_utils import get_experiment_name
from common.log_utils import log_to_sentry
from common.log_utils import logger
from common.utils import compact
from computronium.api import run_commands
from computronium.api import start_commands
from computronium.data_types import Command
from computronium.data_types import Result
from computronium.errors import MetaWorkerFailure
from computronium.errors import NotAllCommandsSucceeded
from computronium.machine_spec import MachineSpec
from computronium.serialization import deserialize
from computronium.serialization import serialize
from computronium.storage_interfaces.etcd_storage import EtcdStorage
from computronium.worker import wait_for_workers
from science.cache.core.const import SCIENCE_BASHENV_SECRETS_FILE_PATH
from science.common.s3_utils import SimpleS3Client

_STATE_KEY_NAME = "/state"


STATE_BUCKET_NAME = "untitled-ai-experiment-iterator-state"

REPO_NAME = "generally_intelligent"
CONTAINER_ROOT_DIR = f"/opt/projects/{REPO_NAME}"

_machine_spec_file = os.path.expanduser("~/machine_spec.txt")
if os.path.exists(_machine_spec_file):
    with open(_machine_spec_file, "r") as infile:
        data = json.loads(infile.read().strip())
        RUNNING_MACHINE_SPEC = MachineSpec(**data)
else:
    RUNNING_MACHINE_SPEC = None  # type: ignore

if os.getenv("IS_EXPERIMENT", "0").lower() in ("1", "true") and RUNNING_MACHINE_SPEC is not None:
    IMAGE_NAME = RUNNING_MACHINE_SPEC.image_name
else:
    IMAGE_NAME = None  # type: ignore

_etcd_storage: Optional[EtcdStorage] = None


@attr.s(auto_attribs=True)
class SerializableCommand:
    """
    The contract of this class is:
    - have only a single method that does not start with an underscore
    - all attributes must be primitives and trivially serializable / deserializable to str
    This enables us to easily turn it into a set of command line arguments
    """


T = TypeVar("T", bound="BaseExperimentState")
CommandType = TypeVar("CommandType", bound=SerializableCommand)


@attr.s(auto_attribs=True)
class BaseExperimentState:
    @classmethod
    def build(cls: Type[T]) -> T:
        raise NotImplementedError()


@attr.s(auto_attribs=True)
class DataExperimentState(BaseExperimentState):
    completed_data_commands: List[str]


class BaseExperiment(Generic[T, CommandType], ABC):
    state_class: Type[T]

    clean: bool = False
    secret_file_path = SCIENCE_BASHENV_SECRETS_FILE_PATH
    is_deterministic: bool = False

    def inner_run(self, state: T):
        raise NotImplementedError

    def run(self):
        if self.clean:
            self._clear_state()
        state = self._load_state()
        self.inner_run(state)

    def _save_state(self, state: T):
        storage = self._get_etcd_storage()
        data = serialize(state)
        storage.store(_STATE_KEY_NAME, data.encode("UTF-8"))

    def _load_state(self) -> T:
        storage = self._get_etcd_storage()
        try:
            data = storage.load(_STATE_KEY_NAME)
        except KeyError:
            return self.state_class.build()
        else:
            return deserialize(data.decode("UTF-8"), self.state_class)

    def _get_relative_experiment_dir(self):
        return f"experiments/{get_experiment_name()}"

    def _get_etcd_storage(self):
        global _etcd_storage
        if _etcd_storage is None:
            _storage = EtcdStorage.build()
            assert _storage is not None, "etcd storage failed to build"
            _etcd_storage = _storage.with_new_prefix(f"/{REPO_NAME}/{self._get_relative_experiment_dir()}")
        return _etcd_storage

    def _clear_state(self):
        self._save_state(self.state_class.build())

    @classmethod
    def get_command(
        cls, command: SerializableCommand, file_name: Optional[str] = None, is_deterministic: bool = False
    ):
        if file_name is None:
            file_name = sys.argv[0]
        remote_experiment_dir = os.getcwd()
        command_name = _get_command_name(command)
        kwargs = _get_command_kwargs(command)
        command_kwarg_string = command_name + " " + " ".join([f"--{k} {v}" for k, v in kwargs.items()])
        deterministic_cuda_string = "CUBLAS_WORKSPACE_CONFIG=:4096:8 " if is_deterministic else None
        pythonpath_string = f"PYTHONPATH=.:{CONTAINER_ROOT_DIR}/computronium:computronium:{CONTAINER_ROOT_DIR}/bones:bones:{CONTAINER_ROOT_DIR}/science:science:$PYTHONPATH"
        python_env_string = "".join(compact([deterministic_cuda_string, pythonpath_string]))
        return f"cd {remote_experiment_dir} && source {cls.secret_file_path} && {python_env_string} python3 -u {file_name} {command_kwarg_string}"

    def is_worth_logging_failure_to_sentry(self, command: Command, result: Result):
        """Override this function to filter out expected failures, ex, OOM when trying large batch sizes"""
        return True

    def log_to_sentry(self, command: Command, result: Result):
        try:
            assert False, "Command failed"
        except AssertionError as e:
            log_to_sentry(e, extra=dict(command=command, result=result))
            raise


DataExperimentType = TypeVar("DataExperimentType", bound="DataExperimentState")


class BaseTwoPhaseDataTrainExperiment(BaseExperiment[DataExperimentType, CommandType]):
    data_worker_count: int
    data_worker_spec: MachineSpec
    train_worker_count: int
    train_worker_spec: MachineSpec

    def get_data_commands(self) -> List[Command]:
        raise NotImplementedError()

    def _train(self, state: DataExperimentType) -> List[Command]:
        raise NotImplementedError()

    def inner_run(self, state: DataExperimentType):
        self._generate_data(state)
        self._train(state)

    def _generate_data(self, state: DataExperimentType):
        data_commands = self.get_data_commands()
        if len(data_commands) > 0:
            completed_commands = set(state.completed_data_commands)
            incomplete_commands = [x for x in data_commands if x not in completed_commands]

            def on_success(command: Command, result: Result):
                state.completed_data_commands.append(command)
                self._save_state(state)

            def on_failure(command: Command, result: Result):
                raise Exception(f"Data generation command failed, cannot continue: {command}\n{result}")

            run_commands(
                iter(incomplete_commands),
                self.data_worker_spec,
                self.data_worker_count,
                self.secret_file_path,
                on_success=on_success,
                on_failure=on_failure,
            )


@attr.s(auto_attribs=True)
class EvalExperimentState(BaseExperimentState):
    train_command_outcomes: Dict[str, bool]
    eval_commands: List[str]
    eval_command_outcomes: Dict[str, bool]

    @classmethod
    def build(cls: Type["EvalExperimentState"]) -> "EvalExperimentState":
        return EvalExperimentState(train_command_outcomes={}, eval_commands=[], eval_command_outcomes={})


EvalExperimentType = TypeVar("EvalExperimentType", bound="EvalExperimentState")


@attr.s(auto_attribs=True)
class TrainEvalIteratorExperimentState(EvalExperimentState):
    serialized_iterator_state: str

    @classmethod
    def build(cls: Type["TrainEvalIteratorExperimentState"]) -> "TrainEvalIteratorExperimentState":
        return TrainEvalIteratorExperimentState(
            train_command_outcomes={}, eval_commands=[], eval_command_outcomes={}, serialized_iterator_state=""
        )


class IteratedTrainEvalExperiment(BaseExperiment[EvalExperimentType, CommandType]):
    state_class: Type[TrainEvalIteratorExperimentState] = TrainEvalIteratorExperimentState
    train_worker_count: int
    train_worker_spec: MachineSpec
    eval_worker_spec: MachineSpec

    def get_train_iterator(self, state: TrainEvalIteratorExperimentState) -> Iterator[CommandType]:
        raise NotImplementedError()

    def get_eval_command(self, train_command: CommandType, result: Result) -> Optional[CommandType]:
        raise NotImplementedError()

    def on_success(self, command: CommandType, result: Result) -> str:
        raise NotImplementedError()

    def on_failure(self, command: CommandType, result: Result) -> str:
        raise NotImplementedError()

    def inner_run(self, state: EvalExperimentType):
        command_lookup: Dict[str, CommandType] = {}
        eval_workers = []
        shared_lock = Lock()

        # TODO: Can we do something better than this...

        def on_eval_success(command: Command, result: Result):
            state.eval_command_outcomes[command] = True
            original_command = command_lookup[command]
            state.serialized_iterator_state = self.on_success(original_command, result)
            self._save_state(state)

        def on_eval_failure(command: Command, result: Result):
            state.eval_command_outcomes[command] = False
            original_command = command_lookup[command]
            state.serialized_iterator_state = self.on_failure(original_command, result)
            self._save_state(state)

        def on_train_success(command: Command, result: Result):
            state.train_command_outcomes[command] = True
            original_command = command_lookup[command]
            state.serialized_iterator_state = self.on_success(original_command, result)
            eval_command = self.get_eval_command(original_command, result)
            if eval_command is None:
                self._save_state(state)
                return
            eval_command_str = self.get_command(eval_command)
            command_lookup[eval_command_str] = eval_command
            state.eval_commands.append(eval_command_str)
            self._save_state(state)
            this_eval_manager, this_eval_workers = start_commands(
                iter([eval_command_str]),
                self.eval_worker_spec,
                1,
                self.secret_file_path,
                on_success=on_eval_success,
                on_failure=on_eval_failure,
                lock=shared_lock,
            )
            eval_workers.extend(this_eval_workers)

        def on_train_failure(command: Command, result: Result):
            state.train_command_outcomes[command] = False
            original_command = command_lookup[command]
            state.serialized_iterator_state = self.on_failure(original_command, result)
            self._save_state(state)

        def inner_iterator():
            for command in self.get_train_iterator(state):
                command_str = self.get_command(command)
                command_lookup[command_str] = command
                logger.info(f"WTF: command lookup: {command_lookup}")
                yield command_str

        train_manager, train_workers = start_commands(
            inner_iterator(),
            self.train_worker_spec,
            self.train_worker_count,
            self.secret_file_path,
            on_success=on_train_success,
            on_failure=on_train_failure,
            lock=shared_lock,
        )

        incomplete_eval_commands = [x for x in state.eval_commands if x not in state.eval_command_outcomes]
        if len(incomplete_eval_commands) > 0:
            incomplete_eval_manager, incomplete_eval_workers = start_commands(
                iter(incomplete_eval_commands),
                self.eval_worker_spec,
                min(self.train_worker_count, len(incomplete_eval_commands)),
                self.secret_file_path,
                on_success=on_train_success,
                on_failure=on_train_failure,
                lock=shared_lock,
            )
            eval_workers.extend(incomplete_eval_workers)

        wait_for_workers(train_workers)
        wait_for_workers(tuple(eval_workers))

        if train_manager.fail_event.is_set():
            raise MetaWorkerFailure("manager or worker encountered an unhandled exception, see above")
        if train_manager.some_command_failed_event.is_set():
            raise NotAllCommandsSucceeded()

        results = tuple(set(state.train_command_outcomes.values()))
        if results == (True,):
            return

        for command, is_success in state.train_command_outcomes.items():
            if not is_success:
                logger.error(f"Train command failed, see above log for reason: {command}")

        for command, is_success in state.eval_command_outcomes.items():
            if not is_success:
                logger.error(f"Eval command failed, see above log for reason: {command}")

        raise Exception("Some commands failed")

    def _save_state(self, state: TrainEvalIteratorExperimentState):
        client = self._get_s3_client()
        data = serialize(state)
        client.save(
            key=f"{self._get_relative_experiment_dir()}/{_STATE_KEY_NAME}",
            data=data.encode("UTF-8"),
        )

    def _load_state(self) -> TrainEvalIteratorExperimentState:
        client = self._get_s3_client()
        try:
            data = client.load(key=f"{self._get_relative_experiment_dir()}/{_STATE_KEY_NAME}")
        except KeyError:
            return self.state_class.build()
        else:
            return deserialize(data.decode("UTF-8"), self.state_class)

    @staticmethod
    def _get_s3_client():
        return SimpleS3Client(STATE_BUCKET_NAME)


@attr.s(auto_attribs=True)
class SimpleExperimentState(DataExperimentState):
    train_command_outcomes: Dict[str, bool]

    @classmethod
    def build(cls: Type["SimpleExperimentState"]) -> "SimpleExperimentState":
        return SimpleExperimentState(completed_data_commands=[], train_command_outcomes={})


@attr.s(auto_attribs=True)
class SimpleTwoPhaseDataTrainExperiment(BaseTwoPhaseDataTrainExperiment[SimpleExperimentState, CommandType]):
    state_class: Type[SimpleExperimentState] = SimpleExperimentState

    def get_train_commands(self) -> List[Command]:
        raise NotImplementedError()

    def _train(self, state: SimpleExperimentState):
        train_commands = self.get_train_commands()
        incomplete_commands = [x for x in train_commands if x not in state.train_command_outcomes]

        def on_success(command: Command, result: Result):
            state.train_command_outcomes[command] = True
            self._save_state(state)

        def on_failure(command: Command, result: Result):
            if self.is_worth_logging_failure_to_sentry(command, result):
                self.log_to_sentry(command, result)
            state.train_command_outcomes[command] = False
            self._save_state(state)

        run_commands(
            iter(incomplete_commands),
            self.train_worker_spec,
            self.train_worker_count,
            self.secret_file_path,
            on_success=on_success,
            on_failure=on_failure,
        )

        results = tuple(set(state.train_command_outcomes.values()))
        if results == (True,):
            return

        for command, is_success in state.train_command_outcomes.items():
            if not is_success:
                logger.error(f"Command failed, see above log for reason: {command}")

        raise Exception("Some commands failed")


@attr.s(auto_attribs=True)
class TrainIteratorExperimentState(DataExperimentState):
    serialized_iterator_state: str

    @classmethod
    def build(cls: Type["TrainIteratorExperimentState"]) -> "TrainIteratorExperimentState":
        return TrainIteratorExperimentState(completed_data_commands=[], serialized_iterator_state="")


@attr.s(auto_attribs=True)
class IteratedTwoPhaseDataTrainExperiment(BaseTwoPhaseDataTrainExperiment[TrainIteratorExperimentState, CommandType]):
    state_class: Type[TrainIteratorExperimentState] = TrainIteratorExperimentState

    def get_train_iterator(self, state: TrainIteratorExperimentState) -> Iterator[CommandType]:
        raise NotImplementedError()

    def on_success(self, command: CommandType, result: Result) -> str:
        raise NotImplementedError()

    def on_failure(self, command: CommandType, result: Result) -> str:
        raise NotImplementedError()

    def _train(self, state: TrainIteratorExperimentState):
        command_lookup: Dict[str, CommandType] = {}

        def on_success(command: Command, result: Result):
            original_command = command_lookup[command]
            state.serialized_iterator_state = self.on_success(original_command, result)
            self._save_state(state)

        def on_failure(command: Command, result: Result):
            if self.is_worth_logging_failure_to_sentry(command, result):
                self.log_to_sentry(command, result)
            original_command = command_lookup[command]
            state.serialized_iterator_state = self.on_failure(original_command, result)
            self._save_state(state)

        def inner_iterator():
            for command in self.get_train_iterator(state):
                command_str = self.get_command(command)
                command_lookup[command_str] = command
                logger.info(f"WTF: command lookup: {command_lookup}")
                yield command_str

        try:
            run_commands(
                inner_iterator(),
                self.train_worker_spec,
                self.train_worker_count,
                self.secret_file_path,
                on_success=on_success,
                on_failure=on_failure,
            )
        except NotAllCommandsSucceeded:
            # don't care about this when we're running as the iterator, it needs to handle that
            pass

    def _save_state(self, state: TrainIteratorExperimentState):
        client = self._get_s3_client()
        data = serialize(state)
        client.save(
            key=f"{self._get_relative_experiment_dir()}/{_STATE_KEY_NAME}",
            data=data.encode("UTF-8"),
        )

    def _load_state(self) -> TrainIteratorExperimentState:
        client = self._get_s3_client()
        try:
            data = client.load(key=f"{self._get_relative_experiment_dir()}/{_STATE_KEY_NAME}")
        except KeyError:
            return self.state_class.build()
        else:
            return deserialize(data.decode("UTF-8"), self.state_class)

    @staticmethod
    def _get_s3_client():
        return SimpleS3Client(STATE_BUCKET_NAME)


def _get_command_name(command: SerializableCommand):
    possible_methods = []
    for key, value in inspect.getmembers(command.__class__, predicate=inspect.isfunction):
        if key.startswith("_"):
            continue
        elif callable(value):
            possible_methods.append(key)
    assert (
        len(possible_methods) == 1
    ), f"Should have exactly one public method, but all of these match: {possible_methods}"
    return possible_methods[0]


def _get_command_kwargs(command: SerializableCommand):
    assert (
        len(list(getattr(command, "__slots__", []))) == 0
    ), "Do not use slots with experiment classes, you will be sad. They break our method for understanding which attributes have changed, which is based on __dict__"
    fields = attr.fields(command.__class__)
    fields_by_name = {x.name: x for x in fields}
    result = {}
    for key, value in command.__dict__.items():
        if key in fields_by_name:
            field = fields_by_name[key]
            if value != field.default:
                assert isinstance(
                    value, (int, float, str, bool)
                ), f"Can only serialize simple types, not {key}={value}"
                result[key] = value
    return result

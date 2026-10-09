import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from umbi.datatypes import Numeric, is_numeric_a_probability

from .entity_space import EntityMapping
from .entity_space_mixins import HasBranchSpace, HasChoiceSpace, HasStateSpace

logger = logging.getLogger(__name__)


def _sorted_order(keys: list) -> list[int]:
    """Indices that sort ``keys`` (stably); cheap if ``keys`` is already sorted."""
    if all(a <= b for a, b in zip(keys, keys[1:])):
        return list(range(len(keys)))
    return sorted(range(len(keys)), key=keys.__getitem__)


def _check_csr(csr: Sequence[int], num_rows: int, name: str) -> None:
    """Check that ``csr`` is a non-decreasing CSR array with ``num_rows`` rows, starting at 0."""
    if len(csr) != num_rows + 1:
        raise ValueError(f"{name} must have length {num_rows + 1}, got {len(csr)}.")
    if csr[0] != 0:
        raise ValueError(f"{name} must start at 0.")
    if any(csr[i] > csr[i + 1] for i in range(num_rows)):
        raise ValueError(f"{name} must be non-decreasing.")


@dataclass
class StateToChoicesMixin(HasStateSpace, HasChoiceSpace):
    #: For every state, a list of choice indices.
    _state_to_choices: EntityMapping[list[int]] = field(init=False)
    #: For every choice, its source state.
    _choice_to_state: EntityMapping[int] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._state_to_choices = EntityMapping(
            name="state_to_choices",
            domain=self._state_space,
            codomain=self._choice_space,
            maps_to_powerset=True,
            default_factory=list,
        )
        self._choice_to_state = EntityMapping(
            name="choice_to_state", domain=self._choice_space, codomain=self._state_space
        )

    @property
    def state_to_choices(self) -> EntityMapping[list[int]]:
        """Get the state-to-choices mapping."""
        return self._state_to_choices

    @property
    def choice_to_state(self) -> EntityMapping[int]:
        """Get the choice-to-state mapping."""
        return self._choice_to_state

    def get_state_choices(self, state: int) -> Sequence[int]:
        """Get the list of choices of the given state."""
        # self._state_space._check_entity(state)
        return self._state_to_choices[state]

    def num_state_choices(self, state: int) -> int:
        """Get the number of choices of the given state."""
        return len(self.get_state_choices(state))


@dataclass
class ChoiceToBranchesMixin(HasChoiceSpace, HasBranchSpace):
    #: For every choice, a list of branch indices.
    _choice_to_branches: EntityMapping[list[int]] = field(init=False)
    #: For every branch, its source choice.
    _branch_to_choice: EntityMapping[int] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._choice_to_branches = EntityMapping(
            name="choice_to_branches",
            domain=self._choice_space,
            codomain=self._branch_space,
            maps_to_powerset=True,
            default_factory=list,
        )
        self._branch_to_choice = EntityMapping(
            name="branch_to_choice", domain=self._branch_space, codomain=self._choice_space
        )

    @property
    def choice_to_branches(self) -> Sequence[list[int]]:
        """Get the choice-to-branches mapping."""
        return self._choice_to_branches

    def get_choice_branches(self, choice: int) -> Sequence[int]:
        """Get the list of branches of the given choice."""
        # self._choice_space._check_entity(choice)
        return self._choice_to_branches[choice]

    def num_choice_branches(self, choice: int) -> int:
        """Get the number of branches of the given choice."""
        return len(self.get_choice_branches(choice))

    def validate(self) -> None:
        self._choice_to_branches.validate()
        self._branch_to_choice.validate()
        super().validate()


@dataclass
class BranchToTargetMixin(HasStateSpace, HasBranchSpace):
    #: For every branch, its target state.
    _branch_to_target: EntityMapping[int] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._branch_to_target = EntityMapping(
            name="branch_to_target", domain=self._branch_space, codomain=self._state_space
        )

    @property
    def branch_to_target(self) -> EntityMapping[int]:
        return self._branch_to_target

    def validate(self) -> None:
        self._branch_to_target.validate()
        super().validate()


@dataclass
class BranchToProbabilityMixin(HasBranchSpace):
    #: For every branch, its probability.
    _branch_to_probability: EntityMapping[Numeric] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self._branch_to_probability = EntityMapping(name="branch_to_probability", domain=self._branch_space)

    @property
    def branch_to_probability(self) -> EntityMapping[Numeric]:
        return self._branch_to_probability

    def validate(self) -> None:
        self._branch_to_probability.validate(allow_undefined_values=True)
        for branch, prob in enumerate(self._branch_to_probability):
            if prob is not None and not is_numeric_a_probability(prob):
                raise ValueError(f"Branch {branch} has invalid probability {prob}.")
        super().validate()


@dataclass
class ChoicesMixin(
    StateToChoicesMixin,
    ChoiceToBranchesMixin,
    BranchToTargetMixin,
    BranchToProbabilityMixin,
):
    def __post_init__(self):
        super().__post_init__()

    def validate(self) -> None:
        branch_to_probability = self._branch_to_probability
        # only look for undefined probabilities per choice if there are any
        has_undefined = any(prob is None for prob in branch_to_probability)
        for choice, branches in enumerate(self._choice_to_branches):
            if len(branches) == 0:
                raise ValueError(f"Choice {choice} has no branches.")
            if not has_undefined:
                continue
            if len(branches) == 1 and branch_to_probability[branches[0]] is None:
                branch_to_probability[branches[0]] = 1
            if any(branch_to_probability[branch] is None for branch in branches):
                raise ValueError(f"Choice {choice} has multiple branches but some have undefined probabilities.")
            # TODO check that probabilities sum to 1
        super().validate()

    def new_state_choice(
        self,
        state: int,
        targets: Sequence[int] | None = None,
        probs: Sequence[Numeric] | None = None,
        target_prob: Callable[[int], Numeric] | None = None,
    ) -> int:
        """Add a choice to the given state.
        :param state: source state for the new choice
        :param targets: optional iterable of target states for the branches of the new choice
        :param probs: optional probabilities for the new branches
        :param target_prob: optional function mapping target states to branch probabilities
        :return: the new choice index
        """
        # self.state_space._check_entity(state)
        choice = self._new_choice()
        self.state_to_choices[state].append(choice)
        self._choice_to_state[choice] = state
        if targets is not None:
            self.new_choice_branches(choice, targets=targets, probs=probs, target_prob=target_prob)
        return choice

    def remove_choice(self, choice: int) -> list[int]:
        """
        Remove a choice and all its branches.
        :param choice: choice to remove
        :return: new-to-old choice mapping after removal
        """
        self._choice_space.check_entity(choice)
        # remove choice from state-to-choices
        state = self._choice_to_state[choice]
        self.state_to_choices[state].remove(choice)
        self._remove_branches(self._choice_to_branches[choice])
        return self._remove_choice(choice)

    def remove_choices(self, choices: Sequence[int]) -> None:
        # copy and sort in reverse to avoid messing up indices when removing
        choices = sorted(choices, reverse=True)
        for choice in choices:
            self.remove_choice(choice)

    def new_choice_branch(
        self,
        choice: int,
        target: int,
        prob: Numeric | None = None,
        target_prob: Callable[[int], Numeric] | None = None,
    ) -> int:
        """Add a branch to the given choice and return it.
        :param choice: source choice for the new branch
        :param target: target state for the new branch
        :param prob: optional probability for the new branch
        :param target_prob: optional function mapping target states to branch probabilities
        :note: if the ATS has branch probabilities, either prob or target_prob must be provided, but not both
        :return: the new branch index
        """
        self._choice_space.check_entity(choice)
        self._state_space.check_entity(target)
        new_branch = self._new_branch()
        self._choice_to_branches[choice].append(new_branch)
        self._branch_to_choice[new_branch] = choice
        self._branch_to_target[new_branch] = target
        if prob is None and target_prob is not None:
            prob = target_prob(target)
        if prob is not None:
            self._branch_to_probability[new_branch] = prob
        return new_branch

    def new_choice_branches(
        self,
        choice: int,
        targets: Sequence[int],
        probs: Sequence[Numeric] | None = None,
        target_prob: Callable[[int], Numeric] | None = None,
    ) -> list[int]:
        """Add branches to the given choice for multiple targets.
        :param choice: choice to add branches to
        :param targets: target states for the new branches
        :param probs: optional probabilities for the new branches
        :param target_prob: optional function mapping target states to branch probabilities
        :note: if the ATS has branch probabilities, either probs or target_prob must be provided, but not both
        :return: list of new branches added
        """
        new_branches = []
        if probs is not None and len(probs) != len(targets):
            raise ValueError("Length of probs must match length of targets.")
        for target_idx, target in enumerate(targets):
            prob = None
            if probs is not None:
                prob = probs[target_idx]
            branch = self.new_choice_branch(choice, target=target, prob=prob, target_prob=target_prob)
            new_branches.append(branch)
        return new_branches

    def new_choices_from_csr(
        self,
        state_to_choices: Sequence[int],
        choice_to_branches: Sequence[int] | None = None,
        branch_to_target: Sequence[int] | None = None,
        branch_to_probability: Sequence[Numeric | None] | None = None,
    ) -> None:
        """Add choices and branches in bulk from CSR arrays, as stored in umbfiles.

        Equivalent to calling ``new_state_choice`` for every choice and
        ``new_choice_branch`` for every branch, in order, but much faster for
        large transition systems.

        :param state_to_choices: CSR of length ``num_states + 1`` starting at 0: the choices of
            state ``s`` are ``state_to_choices[s]`` .. ``state_to_choices[s + 1] - 1`` (offset by the
            number of existing choices)
        :param choice_to_branches: optional CSR of length ``num_choices + 1`` with the branches of every choice
        :param branch_to_target: target state of every branch, required if there are branches
        :param branch_to_probability: optional probability of every branch
        """
        num_states = self._state_space.num_entities
        _check_csr(state_to_choices, num_states, "state_to_choices")
        num_choices = state_to_choices[-1]
        num_branches = 0
        if choice_to_branches is not None:
            _check_csr(choice_to_branches, num_choices, "choice_to_branches")
            num_branches = choice_to_branches[-1]
            if branch_to_target is None or len(branch_to_target) < num_branches:
                raise ValueError("branch_to_target must have a target for every branch.")
            if num_branches > 0 and (min(branch_to_target[:num_branches]) < 0 or max(branch_to_target[:num_branches]) >= num_states):
                raise ValueError(f"Invalid target in branch_to_target, must be in [0, {num_states - 1}].")
            if branch_to_probability is not None and len(branch_to_probability) < num_branches:
                raise ValueError("branch_to_probability must have a probability for every branch.")

        # choices
        first_choice = self._choice_space.num_entities
        self._new_choices(num_choices)
        choice_to_state = []
        for state in range(num_states):
            begin = first_choice + state_to_choices[state]
            end = first_choice + state_to_choices[state + 1]
            if end > begin:
                self._state_to_choices[state].extend(range(begin, end))
                choice_to_state.extend([state] * (end - begin))
        self._choice_to_state[first_choice:] = choice_to_state
        if choice_to_branches is None:
            return

        # branches
        first_branch = self._branch_space.num_entities
        self._new_branches(num_branches)
        branch_to_choice = []
        for choice in range(num_choices):
            begin = first_branch + choice_to_branches[choice]
            end = first_branch + choice_to_branches[choice + 1]
            self._choice_to_branches[first_choice + choice].extend(range(begin, end))
            branch_to_choice.extend([first_choice + choice] * (end - begin))
        self._branch_to_choice[first_branch:] = branch_to_choice
        self._branch_to_target[first_branch:] = branch_to_target[:num_branches]  # type: ignore[index]
        if branch_to_probability is not None:
            self._branch_to_probability[first_branch:] = branch_to_probability[:num_branches]

    def remove_branch(self, branch: int) -> None:
        self._branch_space.check_entity(branch)
        choice = self._branch_to_choice[branch]
        self._choice_to_branches[choice].remove(branch)
        self._remove_branch(branch)

    def remove_branches(self, branches: Sequence[int]) -> None:
        for branch in branches:
            self.remove_branch(branch)

    def sort_choices(self) -> list[int]:
        """Reorder choices by sorting by their source states.
        :return: new-to-old choice mapping after sorting
        """
        new_to_old = _sorted_order(list(self._choice_to_state))
        self._permute_choices(new_to_old)
        for state in self._state_space.entities:
            self.state_to_choices[state] = sorted(self.state_to_choices[state])
        return new_to_old

    def sort_branches(self) -> list[int]:
        """Reorder branches by sorting by their source choices and targets.
        :return: new-to-old branch mapping after sorting
        """
        new_to_old = _sorted_order(list(zip(self._branch_to_choice, self._branch_to_target)))
        self._permute_branches(new_to_old)
        for choice in self._choice_space.entities:
            self._choice_to_branches[choice] = sorted(self._choice_to_branches[choice])
        return new_to_old

    def sort_transitions(self) -> None:
        """Reorder choices and branches by sorting by their source states and choices."""
        self.sort_choices()
        self.sort_branches()

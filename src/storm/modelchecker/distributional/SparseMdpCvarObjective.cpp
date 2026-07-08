#include "storm/modelchecker/distributional/SparseMdpCvarObjective.h"

#include <deque>
#include <limits>
#include <utility>

#include <boost/optional.hpp>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/UnexpectedException.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
SparseMdpCvarObjective<ValueType>::SparseMdpCvarObjective(storm::storage::SparseMatrix<ValueType> const& transitionMatrix,
                                                          std::vector<ValueType> const& stateActionRewards, storm::storage::BitVector const& targetStates,
                                                          storm::storage::BitVector const& properStates, DistributionalValueIterationOptions const& options,
                                                          PreprocessorResult const& preprocessorResult)
    : transitionMatrix(transitionMatrix),
      stateActionRewards(stateActionRewards),
      targetStates(targetStates),
      properStates(properStates),
      options(options),
      viHelper(transitionMatrix, stateActionRewards, targetStates, properStates, options),
      preprocessorResult(preprocessorResult),
      stateCount(transitionMatrix.getRowGroupCount()),
      budgetCount(preprocessorResult.getNumberOfBudgetAtoms()) {
    validateDimensions();
}

template<typename ValueType>
uint64_t SparseMdpCvarObjective<ValueType>::getStateCount() const {
    return stateCount;
}

template<typename ValueType>
uint64_t SparseMdpCvarObjective<ValueType>::getBudgetCount() const {
    return budgetCount;
}

template<typename ValueType>
uint64_t SparseMdpCvarObjective<ValueType>::getProductStateCount() const {
    return stateCount * budgetCount;
}

template<typename ValueType>
uint64_t SparseMdpCvarObjective<ValueType>::getProductStateIndex(uint64_t state, uint64_t budgetIndex) const {
    STORM_LOG_THROW(state < stateCount, storm::exceptions::InvalidArgumentException,
                    "CVaR product state requested original state " << state << ", but the model has " << stateCount << " states.");
    STORM_LOG_THROW(budgetIndex < budgetCount, storm::exceptions::InvalidArgumentException,
                    "CVaR product state requested budget index " << budgetIndex << ", but the budget grid has " << budgetCount << " atoms.");
    return state * budgetCount + budgetIndex;
}

template<typename ValueType>
uint64_t SparseMdpCvarObjective<ValueType>::getOriginalState(uint64_t productState) const {
    STORM_LOG_THROW(productState < getProductStateCount(), storm::exceptions::InvalidArgumentException,
                    "CVaR product state index " << productState << " is out of range for " << getProductStateCount() << " product states.");
    return productState / budgetCount;
}

template<typename ValueType>
uint64_t SparseMdpCvarObjective<ValueType>::getBudgetIndex(uint64_t productState) const {
    STORM_LOG_THROW(productState < getProductStateCount(), storm::exceptions::InvalidArgumentException,
                    "CVaR product state index " << productState << " is out of range for " << getProductStateCount() << " product states.");
    return productState % budgetCount;
}

template<typename ValueType>
bool SparseMdpCvarObjective<ValueType>::isFiniteProductState(uint64_t productState) const {
    return properStates.get(getOriginalState(productState));
}

template<typename ValueType>
bool SparseMdpCvarObjective<ValueType>::isChoiceAdmissible(uint64_t choice) const {
    return viHelper.isChoiceAdmissible(choice);
}

template<typename ValueType>
uint64_t SparseMdpCvarObjective<ValueType>::getChoiceRewardAsInteger(uint64_t choice) const {
    return viHelper.getChoiceRewardAsInteger(choice);
}

template<typename ValueType>
typename SparseMdpCvarObjective<ValueType>::Distribution SparseMdpCvarObjective<ValueType>::buildProductChoiceDistribution(
    ReachableProductStates const& productStates, std::vector<Distribution> const& previousDistributions, uint64_t choice, uint64_t budgetIndex) const {
    RewardDistributionBuilder<ValueType> builder(options.toRewardDistributionOptions());
    uint64_t const reward = getChoiceRewardAsInteger(choice);
    uint64_t const nextBudgetIndex = preprocessorResult.getNextBudgetIndex(budgetIndex, stateActionRewards[choice]);
    for (auto const& entry : transitionMatrix.getRow(choice)) {
        if (storm::utility::isZero(entry.getValue())) {
            continue;
        }
        uint64_t const successorProductState = getProductStateIndex(entry.getColumn(), nextBudgetIndex);
        auto const successorIndex = productStates.indices.find(successorProductState);
        STORM_LOG_THROW(successorIndex != productStates.indices.end(), storm::exceptions::UnexpectedException,
                        "CVaR product value iteration is missing reachable successor product state " << successorProductState << ".");
        builder.addScaledShifted(entry.getValue(), previousDistributions[successorIndex->second], reward);
    }

    return std::move(builder).build();
}

template<typename ValueType>
ValueType SparseMdpCvarObjective<ValueType>::computeTailDistance(Distribution const& distribution, ValueType const& budget) const {
    ValueType result = storm::utility::zero<ValueType>();
    distribution.forEachMass([this, &result, &budget](ValueType const& reward, ValueType const& mass) {
        if (options.cvarInterpretation == DistributionalCvarInterpretation::Cost && reward > budget) {
            result += (reward - budget) * mass;
        } else if (options.cvarInterpretation == DistributionalCvarInterpretation::Reward && reward < budget) {
            result += (budget - reward) * mass;
        }
    });
    return result;
}

template<typename ValueType>
ValueType SparseMdpCvarObjective<ValueType>::computeCvarValue(Distribution const& distribution, ValueType const& budget) const {
    ValueType const tailMass = storm::utility::convertNumber<ValueType>(options.alpha);
    ValueType const tailDistance = computeTailDistance(distribution, budget);
    if (options.cvarInterpretation == DistributionalCvarInterpretation::Reward) {
        return budget - tailDistance / tailMass;
    }
    return budget + tailDistance / tailMass;
}

template<typename ValueType>
bool SparseMdpCvarObjective<ValueType>::isBetterTailDistance(ValueType const& candidate, ValueType const& current) const {
    bool const minimizeTailDistance =
        (options.cvarInterpretation == DistributionalCvarInterpretation::Cost && storm::solver::minimize(options.optimizationDirection)) ||
        (options.cvarInterpretation == DistributionalCvarInterpretation::Reward && storm::solver::maximize(options.optimizationDirection));
    return minimizeTailDistance ? candidate < current : candidate > current;
}

template<typename ValueType>
bool SparseMdpCvarObjective<ValueType>::isBetterCvarValue(ValueType const& candidate, ValueType const& current) const {
    return storm::solver::minimize(options.optimizationDirection) ? candidate < current : candidate > current;
}

template<typename ValueType>
typename SparseMdpCvarObjective<ValueType>::ReachableProductStates SparseMdpCvarObjective<ValueType>::computeReachableProductStates() const {
    ReachableProductStates result;
    result.indicesByState.resize(stateCount);
    std::deque<uint64_t> worklist;
    result.indices.reserve(budgetCount);

    auto addProductState = [this, &result, &worklist](uint64_t productState) {
        auto const inserted = result.indices.emplace(productState, result.states.size());
        if (inserted.second) {
            result.states.push_back(productState);
            result.indicesByState[getOriginalState(productState)].push_back(inserted.first->second);
            worklist.push_back(productState);
        }
    };

    uint64_t const initialState = preprocessorResult.initialState;
    STORM_LOG_THROW(properStates.get(initialState), storm::exceptions::InvalidArgumentException,
                    "Cannot run CVaR product value iteration because initial state " << initialState << " is not proper.");
    for (uint64_t budgetIndex = 0; budgetIndex < budgetCount; ++budgetIndex) {
        addProductState(getProductStateIndex(initialState, budgetIndex));
    }

    while (!worklist.empty()) {
        uint64_t const productState = worklist.front();
        worklist.pop_front();
        uint64_t const state = getOriginalState(productState);
        uint64_t const budgetIndex = getBudgetIndex(productState);
        STORM_LOG_THROW(properStates.get(state), storm::exceptions::InvalidArgumentException,
                        "Cannot run CVaR product value iteration because reachable original state " << state << " is not proper.");
        if (targetStates.get(state)) {
            continue;
        }
        for (auto const choice : transitionMatrix.getRowGroupIndices(state)) {
            if (!isChoiceAdmissible(choice)) {
                continue;
            }
            uint64_t const nextBudgetIndex = preprocessorResult.getNextBudgetIndex(budgetIndex, stateActionRewards[choice]);
            for (auto const& entry : transitionMatrix.getRow(choice)) {
                if (!storm::utility::isZero(entry.getValue())) {
                    addProductState(getProductStateIndex(entry.getColumn(), nextBudgetIndex));
                }
            }
        }
    }

    return result;
}

template<typename ValueType>
void SparseMdpCvarObjective<ValueType>::runTopologicalViSweep(ReachableProductStates const& productStates, std::vector<Distribution>& distributions) const {
    for (auto stateIt = preprocessorResult.topologicalOrder.rbegin(); stateIt != preprocessorResult.topologicalOrder.rend(); ++stateIt) {
        uint64_t const state = *stateIt;
        for (auto const productStateIndex : productStates.indicesByState[state]) {
            uint64_t const productState = productStates.states[productStateIndex];
            uint64_t const budgetIndex = getBudgetIndex(productState);
            ValueType const& budget = preprocessorResult.getBudgetValue(budgetIndex);

            boost::optional<Distribution> bestDistribution;
            ValueType bestTailExpectation = storm::utility::zero<ValueType>();
            for (auto const choice : transitionMatrix.getRowGroupIndices(state)) {
                if (!isChoiceAdmissible(choice)) {
                    continue;
                }
                Distribution choiceDistribution = buildProductChoiceDistribution(productStates, distributions, choice, budgetIndex);
                ValueType const choiceTailExpectation = computeTailDistance(choiceDistribution, budget);
                if (!bestDistribution || isBetterTailDistance(choiceTailExpectation, bestTailExpectation)) {
                    bestTailExpectation = choiceTailExpectation;
                    bestDistribution = std::move(choiceDistribution);
                }
            }

            STORM_LOG_THROW(bestDistribution, storm::exceptions::UnexpectedException,
                            "Expected at least one admissible CVaR product choice for original state " << state << ".");
            distributions[productStateIndex] = std::move(bestDistribution.get());
        }
    }
}

template<typename ValueType>
typename SparseMdpCvarObjective<ValueType>::Result SparseMdpCvarObjective<ValueType>::selectInitialDistribution(
    ReachableProductStates const& productStates, std::vector<Distribution> const& distributions) const {
    uint64_t const initialState = preprocessorResult.initialState;

    boost::optional<Distribution> selectedDistribution;
    ValueType bestCvarValue = storm::utility::zero<ValueType>();
    for (uint64_t budgetIndex = 0; budgetIndex < budgetCount; ++budgetIndex) {
        uint64_t const productState = getProductStateIndex(initialState, budgetIndex);
        auto const productStateIndex = productStates.indices.find(productState);
        STORM_LOG_THROW(productStateIndex != productStates.indices.end(), storm::exceptions::UnexpectedException,
                        "CVaR initial budget product state " << productState << " is missing from the reachable product state set.");
        Distribution const& distribution = distributions[productStateIndex->second];
        ValueType const& budget = preprocessorResult.getBudgetValue(budgetIndex);
        ValueType const cvarValue = computeCvarValue(distribution, budget);
        if (!selectedDistribution || isBetterCvarValue(cvarValue, bestCvarValue)) {
            bestCvarValue = cvarValue;
            selectedDistribution = distribution;
        }
    }

    STORM_LOG_THROW(selectedDistribution, storm::exceptions::UnexpectedException,
                    "Expected at least one initial CVaR budget candidate for state " << initialState << ".");

    DistributionMap result;
    result.emplace(static_cast<storm::storage::sparse::state_type>(initialState), std::move(selectedDistribution.get()));
    return Result{std::move(result)};
}

template<typename ValueType>
typename SparseMdpCvarObjective<ValueType>::Result SparseMdpCvarObjective<ValueType>::computeCvarOptimalDistribution() const {
    ReachableProductStates const productStates = computeReachableProductStates();
    STORM_LOG_THROW(!productStates.states.empty(), storm::exceptions::UnexpectedException,
                    "Expected at least one reachable CVaR product state for the topological value-iteration sweep.");

    std::vector<Distribution> distributions;
    distributions.reserve(productStates.states.size());
    RewardDistributionOptions const rewardDistributionOptions = options.toRewardDistributionOptions();
    for (auto const productState : productStates.states) {
        uint64_t const state = getOriginalState(productState);
        distributions.push_back(targetStates.get(state) ? Distribution::pointMass(0) : Distribution::categoricalTail(rewardDistributionOptions));
    }

    runTopologicalViSweep(productStates, distributions);
    return selectInitialDistribution(productStates, distributions);
}

template<typename ValueType>
void SparseMdpCvarObjective<ValueType>::validateDimensions() const {
    STORM_LOG_THROW(transitionMatrix.getRowCount() == stateActionRewards.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one normalized state-action reward per nondeterministic choice, but got "
                        << stateActionRewards.size() << " rewards for " << transitionMatrix.getRowCount() << " choices.");
    STORM_LOG_THROW(stateCount == targetStates.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one target-state bit per state, but got a vector of size " << targetStates.size() << " for " << stateCount
                                                                                                               << " states.");
    STORM_LOG_THROW(stateCount == properStates.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one proper-state bit per state, but got a vector of size " << properStates.size() << " for " << stateCount
                                                                                                               << " states.");
    STORM_LOG_THROW(stateCount == preprocessorResult.lowerRewardBounds.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one lower reward bound per state, but got " << preprocessorResult.lowerRewardBounds.size() << " bounds for "
                                                                                                << stateCount << " states.");
    STORM_LOG_THROW(stateCount == preprocessorResult.upperRewardBounds.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one upper reward bound per state, but got " << preprocessorResult.upperRewardBounds.size() << " bounds for "
                                                                                                << stateCount << " states.");
    STORM_LOG_THROW(stateCount == preprocessorResult.finiteRewardStates.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one finite-reward-state bit per state, but got a vector of size "
                        << preprocessorResult.finiteRewardStates.size() << " for " << stateCount << " states.");
    STORM_LOG_THROW(preprocessorResult.initialState < stateCount, storm::exceptions::InvalidArgumentException,
                    "CVaR product objective received initial state " << preprocessorResult.initialState << ", but the model has " << stateCount << " states.");
    STORM_LOG_THROW(budgetCount > 0, storm::exceptions::InvalidArgumentException, "CVaR product objective requires a non-empty budget grid.");
    STORM_LOG_THROW(stateCount <= std::numeric_limits<uint64_t>::max() / budgetCount, storm::exceptions::InvalidArgumentException,
                    "CVaR product objective state space size overflows uint64_t for " << stateCount << " states and " << budgetCount << " budget atoms.");

    storm::storage::BitVector topologicalStates(stateCount, false);
    for (auto const state : preprocessorResult.topologicalOrder) {
        STORM_LOG_THROW(state < stateCount, storm::exceptions::InvalidArgumentException,
                        "CVaR product objective received topological-order state " << state << ", but the model has " << stateCount << " states.");
        STORM_LOG_THROW(properStates.get(state) && !targetStates.get(state), storm::exceptions::InvalidArgumentException,
                        "CVaR product objective topological order contains state " << state << ", but the topological order should contain only "
                                                                                   << "proper non-target states.");
        STORM_LOG_THROW(!topologicalStates.get(state), storm::exceptions::InvalidArgumentException,
                        "CVaR product objective topological order contains state " << state << " more than once.");
        topologicalStates.set(state, true);
    }
    for (uint64_t state = 0; state < stateCount; ++state) {
        if (properStates.get(state) && !targetStates.get(state)) {
            STORM_LOG_THROW(topologicalStates.get(state), storm::exceptions::InvalidArgumentException,
                            "CVaR product objective topological order is missing proper non-target state " << state << ".");
        }
    }
}

template class SparseMdpCvarObjective<double>;
template class SparseMdpCvarObjective<storm::RationalNumber>;

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

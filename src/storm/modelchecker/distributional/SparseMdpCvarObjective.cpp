#include "storm/modelchecker/distributional/SparseMdpCvarObjective.h"

#include <limits>
#include <utility>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
SparseMdpCvarObjective<ValueType>::SparseMdpCvarObjective(
    storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& stateActionRewards,
    storm::storage::BitVector const& targetStates, storm::storage::BitVector const& properStates, DistributionalValueIterationOptions const& options,
    PreprocessorResult const& preprocessorResult)
    : transitionMatrix(transitionMatrix),
      stateActionRewards(stateActionRewards),
      targetStates(targetStates),
      properStates(properStates),
      options(options),
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
typename SparseMdpCvarObjective<ValueType>::ProductDistributionCache SparseMdpCvarObjective<ValueType>::createProductDistributionCache() const {
    return ProductDistributionCache();
}

template<typename ValueType>
typename SparseMdpCvarObjective<ValueType>::Distribution const& SparseMdpCvarObjective<ValueType>::getOrInitializeProductDistribution(
    ProductDistributionCache& cache, uint64_t productState) const {
    uint64_t const state = getOriginalState(productState);
    auto const cached = cache.distributions.find(productState);
    if (cached != cache.distributions.end()) {
        return cached->second;
    }

    STORM_LOG_THROW(properStates.get(state), storm::exceptions::InvalidArgumentException,
                    "Cannot initialize a CVaR product distribution for product state " << productState << " because original state " << state
                                                                                       << " is not proper.");
    Distribution distribution =
        targetStates.get(state) ? Distribution::pointMass(0) : Distribution::categoricalTail(options.toRewardDistributionOptions());
    auto const inserted = cache.distributions.emplace(productState, std::move(distribution));
    return inserted.first->second;
}

template<typename ValueType>
void SparseMdpCvarObjective<ValueType>::validateDimensions() const {
    STORM_LOG_THROW(transitionMatrix.getRowCount() == stateActionRewards.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one normalized state-action reward per nondeterministic choice, but got "
                        << stateActionRewards.size() << " rewards for " << transitionMatrix.getRowCount() << " choices.");
    STORM_LOG_THROW(stateCount == targetStates.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one target-state bit per state, but got a vector of size " << targetStates.size() << " for "
                                                                                                              << stateCount << " states.");
    STORM_LOG_THROW(stateCount == properStates.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one proper-state bit per state, but got a vector of size " << properStates.size() << " for "
                                                                                                              << stateCount << " states.");
    STORM_LOG_THROW(stateCount == preprocessorResult.lowerRewardBounds.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one lower reward bound per state, but got " << preprocessorResult.lowerRewardBounds.size()
                                                                                               << " bounds for " << stateCount << " states.");
    STORM_LOG_THROW(stateCount == preprocessorResult.upperRewardBounds.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one upper reward bound per state, but got " << preprocessorResult.upperRewardBounds.size()
                                                                                               << " bounds for " << stateCount << " states.");
    STORM_LOG_THROW(stateCount == preprocessorResult.finiteRewardStates.size(), storm::exceptions::InvalidArgumentException,
                    "CVaR product objective expects one finite-reward-state bit per state, but got a vector of size "
                        << preprocessorResult.finiteRewardStates.size() << " for " << stateCount << " states.");
    STORM_LOG_THROW(preprocessorResult.initialState < stateCount, storm::exceptions::InvalidArgumentException,
                    "CVaR product objective received initial state " << preprocessorResult.initialState << ", but the model has " << stateCount
                                                                    << " states.");
    STORM_LOG_THROW(budgetCount > 0, storm::exceptions::InvalidArgumentException, "CVaR product objective requires a non-empty budget grid.");
    STORM_LOG_THROW(stateCount <= std::numeric_limits<uint64_t>::max() / budgetCount, storm::exceptions::InvalidArgumentException,
                    "CVaR product objective state space size overflows uint64_t for " << stateCount << " states and " << budgetCount
                                                                                     << " budget atoms.");
}

template class SparseMdpCvarObjective<double>;
template class SparseMdpCvarObjective<storm::RationalNumber>;

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/exceptions/UnexpectedException.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/utility/constants.h"
#include "storm/utility/graph.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename ValueType>
class SparseMdpCvarPreprocessor {
   public:
    struct Result {
        std::vector<ValueType> lowerRewardBounds;
        std::vector<ValueType> upperRewardBounds;
        storm::storage::BitVector finiteRewardStates;
        std::vector<uint64_t> topologicalOrder;
        uint64_t initialState;
        ValueType initialLowerRewardBound;
        ValueType initialUpperRewardBound;
        std::vector<ValueType> budgetGrid;

        uint64_t getNumberOfBudgetAtoms() const {
            return budgetGrid.size();
        }

        ValueType const& getBudgetValue(uint64_t budgetIndex) const {
            STORM_LOG_THROW(budgetIndex < budgetGrid.size(), storm::exceptions::InvalidArgumentException,
                            "CVaR budget index " << budgetIndex << " is out of range for a grid with " << budgetGrid.size() << " atoms.");
            return budgetGrid[budgetIndex];
        }

        uint64_t getFirstInitialBudgetIndex() const {
            auto const firstInitialBudget = std::lower_bound(budgetGrid.begin(), budgetGrid.end(), initialLowerRewardBound);
            STORM_LOG_THROW(firstInitialBudget != budgetGrid.end() && *firstInitialBudget == initialLowerRewardBound,
                            storm::exceptions::UnexpectedException,
                            "CVaR residual budget grid does not contain the lower endpoint of the initial threshold interval.");
            return static_cast<uint64_t>(firstInitialBudget - budgetGrid.begin());
        }

        uint64_t getNextBudgetIndex(uint64_t currentBudgetIndex, ValueType const& reward) const {
            STORM_LOG_THROW(currentBudgetIndex < budgetGrid.size(), storm::exceptions::InvalidArgumentException,
                            "CVaR current budget index " << currentBudgetIndex << " is out of range for a grid with " << budgetGrid.size() << " atoms.");
            STORM_LOG_THROW(reward >= storm::utility::zero<ValueType>() && storm::utility::isInteger(reward), storm::exceptions::InvalidArgumentException,
                            "CVaR budget transitions require a non-negative integer reward, but got " << reward << ".");

            ValueType const nextBudget = budgetGrid[currentBudgetIndex] - reward;
            auto const upper = std::upper_bound(budgetGrid.begin(), budgetGrid.end(), nextBudget);
            if (upper == budgetGrid.begin()) {
                return 0;
            }
            return static_cast<uint64_t>((upper - budgetGrid.begin()) - 1);
        }
    };

    SparseMdpCvarPreprocessor(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& stateActionRewards,
                              storm::storage::BitVector const& targetStates, storm::storage::BitVector const& properStates, uint64_t initialState)
        : transitionMatrix(transitionMatrix),
          stateActionRewards(stateActionRewards),
          targetStates(targetStates),
          properStates(properStates),
          properNonTargetStates(properStates & ~targetStates),
          initialState(initialState) {
        STORM_LOG_THROW(transitionMatrix.getRowCount() == stateActionRewards.size(), storm::exceptions::InvalidArgumentException,
                        "CVaR preprocessing expects one normalized state-action reward per nondeterministic choice, but got "
                            << stateActionRewards.size() << " rewards for " << transitionMatrix.getRowCount() << " choices.");
        STORM_LOG_THROW(transitionMatrix.getRowGroupCount() == targetStates.size(), storm::exceptions::InvalidArgumentException,
                        "CVaR preprocessing expects one target-state bit per state, but got a vector of size "
                            << targetStates.size() << " for " << transitionMatrix.getRowGroupCount() << " states.");
        STORM_LOG_THROW(transitionMatrix.getRowGroupCount() == properStates.size(), storm::exceptions::InvalidArgumentException,
                        "CVaR preprocessing expects one proper-state bit per state, but got a vector of size "
                            << properStates.size() << " for " << transitionMatrix.getRowGroupCount() << " states.");
        STORM_LOG_THROW(
            initialState < transitionMatrix.getRowGroupCount(), storm::exceptions::InvalidArgumentException,
            "CVaR preprocessing received initial state " << initialState << ", but the model has " << transitionMatrix.getRowGroupCount() << " states.");
        STORM_LOG_THROW(properStates.full(), storm::exceptions::NotSupportedException,
                        "CVaR preprocessing currently requires every state to be proper, i.e., every state must admit an almost-sure target-reaching "
                        "scheduler. The first CVaR bounded-support contract only computes finite support bounds for fully proper models, but got "
                            << properStates.getNumberOfSetBits() << " proper states out of " << properStates.size() << ".");
    }

    Result computeRewardBounds() const {
        STORM_LOG_THROW(!storm::utility::graph::hasCycle(transitionMatrix, properNonTargetStates), storm::exceptions::NotSupportedException,
                        "CVaR distributional value iteration currently requires an acyclic proper non-target subsystem.");

        Result result;
        result.lowerRewardBounds.resize(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
        result.upperRewardBounds.resize(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
        result.finiteRewardStates = properStates;
        result.topologicalOrder = computeTopologicalOrder();
        result.initialState = initialState;

        for (auto it = result.topologicalOrder.rbegin(); it != result.topologicalOrder.rend(); ++it) {
            uint64_t const state = *it;
            bool foundAdmissibleChoice = false;
            ValueType stateLowerBound = storm::utility::zero<ValueType>();
            ValueType stateUpperBound = storm::utility::zero<ValueType>();
            for (auto const choice : transitionMatrix.getRowGroupIndices(state)) {
                if (!choiceStaysInProperStates(choice)) {
                    continue;
                }
                auto const choiceBounds = computeChoiceBounds(choice, result.lowerRewardBounds, result.upperRewardBounds);
                if (!foundAdmissibleChoice) {
                    stateLowerBound = choiceBounds.first;
                    stateUpperBound = choiceBounds.second;
                    foundAdmissibleChoice = true;
                } else {
                    stateLowerBound = std::min(stateLowerBound, choiceBounds.first);
                    stateUpperBound = std::max(stateUpperBound, choiceBounds.second);
                }
            }
            STORM_LOG_THROW(foundAdmissibleChoice, storm::exceptions::UnexpectedException,
                            "Expected at least one admissible CVaR distributional choice for a proper state.");
            result.lowerRewardBounds[state] = stateLowerBound;
            result.upperRewardBounds[state] = stateUpperBound;
        }

        result.initialLowerRewardBound = result.lowerRewardBounds[initialState];
        result.initialUpperRewardBound = result.upperRewardBounds[initialState];
        result.budgetGrid = computeBudgetGrid(storm::utility::zero<ValueType>(), result.initialUpperRewardBound);
        STORM_LOG_INFO("CVaR initial threshold interval for state " << initialState << " spans [" << result.initialLowerRewardBound << ", "
                                                                    << result.initialUpperRewardBound << "]; the residual budget grid spans [0, "
                                                                    << result.initialUpperRewardBound << "] with " << result.budgetGrid.size()
                                                                    << " exact integer threshold(s).");

        return result;
    }

   private:
    std::vector<ValueType> computeBudgetGrid(ValueType const& lowerBound, ValueType const& upperBound) const {
        STORM_LOG_THROW(storm::utility::isInteger(lowerBound) && storm::utility::isInteger(upperBound), storm::exceptions::UnexpectedException,
                        "Expected integer CVaR reward bounds, but got [" << lowerBound << ", " << upperBound << "].");
        STORM_LOG_THROW(lowerBound <= upperBound, storm::exceptions::UnexpectedException,
                        "Expected ordered CVaR reward bounds, but got [" << lowerBound << ", " << upperBound << "].");
        STORM_LOG_THROW(lowerBound >= storm::utility::zero<ValueType>(), storm::exceptions::UnexpectedException,
                        "Expected non-negative CVaR reward bounds, but got [" << lowerBound << ", " << upperBound << "].");

        uint64_t const lowerBoundAsInteger = storm::utility::convertNumber<uint64_t, ValueType>(lowerBound);
        uint64_t const upperBoundAsInteger = storm::utility::convertNumber<uint64_t, ValueType>(upperBound);
        STORM_LOG_THROW(lowerBoundAsInteger <= upperBoundAsInteger, storm::exceptions::UnexpectedException,
                        "Expected ordered integer CVaR reward bounds, but got [" << lowerBoundAsInteger << ", " << upperBoundAsInteger << "].");

        uint64_t const gridSize = upperBoundAsInteger - lowerBoundAsInteger + 1;
        std::vector<ValueType> result;
        result.reserve(gridSize);
        for (uint64_t offset = 0; offset < gridSize; ++offset) {
            result.push_back(storm::utility::convertNumber<ValueType, uint64_t>(lowerBoundAsInteger + offset));
        }

        STORM_LOG_THROW(result.front() == lowerBound && result.back() == upperBound, storm::exceptions::UnexpectedException,
                        "Failed to construct a CVaR budget grid that includes the requested threshold endpoints.");
        return result;
    }

    bool choiceStaysInProperStates(uint64_t choice) const {
        for (auto const& entry : transitionMatrix.getRow(choice)) {
            if (!storm::utility::isZero(entry.getValue()) && !targetStates.get(entry.getColumn()) && !properStates.get(entry.getColumn())) {
                return false;
            }
        }
        return true;
    }

    std::pair<ValueType, ValueType> computeChoiceBounds(uint64_t choice, std::vector<ValueType> const& lowerRewardBounds,
                                                        std::vector<ValueType> const& upperRewardBounds) const {
        ValueType successorLowerBound = storm::utility::zero<ValueType>();
        ValueType successorUpperBound = storm::utility::zero<ValueType>();
        bool foundSuccessor = false;
        for (auto const& entry : transitionMatrix.getRow(choice)) {
            if (storm::utility::isZero(entry.getValue())) {
                continue;
            }
            ValueType currentLowerBound = storm::utility::zero<ValueType>();
            ValueType currentUpperBound = storm::utility::zero<ValueType>();
            if (!targetStates.get(entry.getColumn())) {
                STORM_LOG_THROW(properNonTargetStates.get(entry.getColumn()), storm::exceptions::UnexpectedException,
                                "Encountered an inadmissible successor while computing CVaR reward bounds.");
                currentLowerBound = lowerRewardBounds[entry.getColumn()];
                currentUpperBound = upperRewardBounds[entry.getColumn()];
            }
            if (!foundSuccessor) {
                successorLowerBound = currentLowerBound;
                successorUpperBound = currentUpperBound;
                foundSuccessor = true;
            } else {
                successorLowerBound = std::min(successorLowerBound, currentLowerBound);
                successorUpperBound = std::max(successorUpperBound, currentUpperBound);
            }
        }

        ValueType const reward = stateActionRewards[choice];
        return {reward + successorLowerBound, reward + successorUpperBound};
    }

    std::vector<uint64_t> computeTopologicalOrder() const {
        std::vector<uint64_t> incomingEdges(transitionMatrix.getRowGroupCount(), 0);
        for (auto const state : properNonTargetStates) {
            for (auto const& entry : transitionMatrix.getRowGroup(state)) {
                if (!storm::utility::isZero(entry.getValue()) && properNonTargetStates.get(entry.getColumn())) {
                    ++incomingEdges[entry.getColumn()];
                }
            }
        }

        std::deque<uint64_t> worklist;
        for (auto const state : properNonTargetStates) {
            if (incomingEdges[state] == 0) {
                worklist.push_back(state);
            }
        }

        std::vector<uint64_t> result;
        result.reserve(properNonTargetStates.getNumberOfSetBits());
        while (!worklist.empty()) {
            uint64_t const state = worklist.front();
            worklist.pop_front();
            result.push_back(state);
            for (auto const& entry : transitionMatrix.getRowGroup(state)) {
                if (!storm::utility::isZero(entry.getValue()) && properNonTargetStates.get(entry.getColumn())) {
                    --incomingEdges[entry.getColumn()];
                    if (incomingEdges[entry.getColumn()] == 0) {
                        worklist.push_back(entry.getColumn());
                    }
                }
            }
        }

        STORM_LOG_THROW(result.size() == properNonTargetStates.getNumberOfSetBits(), storm::exceptions::UnexpectedException,
                        "Failed to compute a topological order for the CVaR distributional proper non-target subsystem.");
        return result;
    }

    storm::storage::SparseMatrix<ValueType> const& transitionMatrix;
    std::vector<ValueType> const& stateActionRewards;
    storm::storage::BitVector targetStates;
    storm::storage::BitVector properStates;
    storm::storage::BitVector properNonTargetStates;
    uint64_t initialState;
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

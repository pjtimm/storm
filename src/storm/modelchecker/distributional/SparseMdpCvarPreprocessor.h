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
    };

    SparseMdpCvarPreprocessor(storm::storage::SparseMatrix<ValueType> const& transitionMatrix, std::vector<ValueType> const& stateActionRewards,
                              storm::storage::BitVector const& targetStates, storm::storage::BitVector const& properStates)
        : transitionMatrix(transitionMatrix),
          stateActionRewards(stateActionRewards),
          targetStates(targetStates),
          properStates(properStates),
          properNonTargetStates(properStates & ~targetStates) {
        STORM_LOG_THROW(transitionMatrix.getRowCount() == stateActionRewards.size(), storm::exceptions::InvalidArgumentException,
                        "Expected one CVaR distributional reward value per nondeterministic choice.");
        STORM_LOG_THROW(transitionMatrix.getRowGroupCount() == targetStates.size(), storm::exceptions::InvalidArgumentException,
                        "CVaR distributional target-state vector has unexpected size.");
        STORM_LOG_THROW(transitionMatrix.getRowGroupCount() == properStates.size(), storm::exceptions::InvalidArgumentException,
                        "CVaR distributional proper-state vector has unexpected size.");
    }

    Result computeRewardBounds() const {
        STORM_LOG_THROW(!storm::utility::graph::hasCycle(transitionMatrix, properNonTargetStates), storm::exceptions::NotSupportedException,
                        "CVaR distributional value iteration currently requires an acyclic proper non-target subsystem.");

        Result result;
        result.lowerRewardBounds.resize(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
        result.upperRewardBounds.resize(transitionMatrix.getRowGroupCount(), storm::utility::zero<ValueType>());
        result.finiteRewardStates = properStates;
        result.topologicalOrder = computeTopologicalOrder();

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

        return result;
    }

   private:
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
        bool foundNonTargetSuccessor = false;
        for (auto const& entry : transitionMatrix.getRow(choice)) {
            if (storm::utility::isZero(entry.getValue()) || targetStates.get(entry.getColumn())) {
                continue;
            }
            STORM_LOG_THROW(properNonTargetStates.get(entry.getColumn()), storm::exceptions::UnexpectedException,
                            "Encountered an inadmissible successor while computing CVaR reward bounds.");
            if (!foundNonTargetSuccessor) {
                successorLowerBound = lowerRewardBounds[entry.getColumn()];
                successorUpperBound = upperRewardBounds[entry.getColumn()];
                foundNonTargetSuccessor = true;
            } else {
                successorLowerBound = std::min(successorLowerBound, lowerRewardBounds[entry.getColumn()]);
                successorUpperBound = std::max(successorUpperBound, upperRewardBounds[entry.getColumn()]);
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
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

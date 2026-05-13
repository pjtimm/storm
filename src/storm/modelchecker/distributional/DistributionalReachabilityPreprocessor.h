#pragma once

#include <memory>
#include <string>

#include "storm/adapters/IntervalForward.h"
#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/logic/Formulas.h"
#include "storm/modelchecker/distributional/DistributionalRewardReachabilityQuery.h"
#include "storm/modelchecker/propositional/SparsePropositionalModelChecker.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/StronglyConnectedComponentDecomposition.h"
#include "storm/utility/constants.h"
#include "storm/utility/macros.h"

namespace storm {
class Environment;

namespace modelchecker {
namespace distributional {

template<class SparseMdpModelType>
struct DistributionalReachabilityPreprocessorResult {
    using ValueType = typename SparseMdpModelType::ValueType;
    using RewardModelType = typename SparseMdpModelType::RewardModelType;

    std::string rewardModelName;
    RewardModelType const* rewardModel;
    std::shared_ptr<storm::logic::Formula const> targetFormula;
    storm::storage::BitVector targetStates;
    std::vector<ValueType> stateActionRewards;
};

template<class SparseMdpModelType>
class DistributionalReachabilityPreprocessor {
   public:
    using ValueType = typename SparseMdpModelType::ValueType;
    using SolutionType = storm::IntervalBaseType<ValueType>;
    using Result = DistributionalReachabilityPreprocessorResult<SparseMdpModelType>;

    static Result preprocess(Environment const& env, SparseMdpModelType const& model, DistributionalRewardReachabilityQuery const& query,
                             bool /*produceScheduler*/) {
        storm::storage::BitVector targetStates = computeTargetStates(env, model, query.targetFormula);
        STORM_LOG_THROW(!targetStates.empty(), storm::exceptions::InvalidPropertyException, "Distributional model checking requires a non-empty target set.");
        STORM_LOG_THROW(targetStates.size() == model.getNumberOfStates(), storm::exceptions::InvalidPropertyException,
                        "Distributional target state vector does not match the model size.");

        std::string rewardModelName = getRewardModelName(model, query.rewardOperatorFormula);
        auto const& rewardModel = model.getRewardModel(rewardModelName);
        STORM_LOG_THROW(!rewardModel.hasTransitionRewards(), storm::exceptions::NotSupportedException,
                        "Distributional model checking does not support transition rewards yet.");
        std::vector<ValueType> stateActionRewards = rewardModel.getTotalRewardVector(model.getTransitionMatrix());
        validateStateActionRewards(stateActionRewards);
        validatePositiveRewardsAreAcyclic(model, targetStates, stateActionRewards);

        return Result{rewardModelName, &rewardModel, query.targetFormula.asSharedPointer(), std::move(targetStates), std::move(stateActionRewards)};
    }

   private:
    static void validateStateActionRewards(std::vector<ValueType> const& stateActionRewards) {
        for (auto const& reward : stateActionRewards) {
            STORM_LOG_THROW(reward >= storm::utility::zero<ValueType>(), storm::exceptions::NotSupportedException,
                            "Distributional model checking currently supports only non-negative rewards.");
            STORM_LOG_THROW(storm::utility::isInteger(reward), storm::exceptions::NotSupportedException,
                            "Distributional model checking currently supports only integer rewards.");
        }
    }

    static void validatePositiveRewardsAreAcyclic(SparseMdpModelType const& model, storm::storage::BitVector const& targetStates,
                                                  std::vector<ValueType> const& stateActionRewards) {
        auto const& transitionMatrix = model.getTransitionMatrix();
        storm::storage::BitVector nonTargetStates = ~targetStates;
        storm::storage::StronglyConnectedComponentDecompositionOptions options;
        options.subsystem(nonTargetStates);
        storm::storage::SccDecompositionResult sccResult;
        storm::storage::performSccDecomposition(transitionMatrix, options, sccResult);

        for (uint64_t state = 0; state < model.getNumberOfStates(); ++state) {
            if (targetStates.get(state) || !sccResult.nonTrivialStates.get(state)) {
                continue;
            }

            auto const stateScc = sccResult.stateToSccMapping[state];
            for (auto const choice : transitionMatrix.getRowGroupIndices(state)) {
                if (storm::utility::isZero(stateActionRewards[choice])) {
                    continue;
                }
                for (auto const& transition : transitionMatrix.getRow(choice)) {
                    if (targetStates.get(transition.getColumn())) {
                        continue;
                    }
                    if (sccResult.stateToSccMapping[transition.getColumn()] == stateScc) {
                        STORM_LOG_THROW(false, storm::exceptions::NotSupportedException,
                                        "Distributional model checking currently supports only DAGs or zero-cost cycles.");
                    }
                }
            }
        }
    }

    static storm::storage::BitVector computeTargetStates(Environment const& env, SparseMdpModelType const& model, storm::logic::Formula const& targetFormula) {
        storm::modelchecker::SparsePropositionalModelChecker<SparseMdpModelType> modelChecker(model);
        auto targetResult = modelChecker.check(env, targetFormula);
        return targetResult->template asExplicitQualitativeCheckResult<SolutionType>().getTruthValuesVector();
    }

    static std::string getRewardModelName(SparseMdpModelType const& model, storm::logic::RewardOperatorFormula const& formula) {
        if (formula.hasRewardModelName()) {
            STORM_LOG_THROW(model.hasRewardModel(formula.getRewardModelName()), storm::exceptions::InvalidPropertyException,
                            "The requested reward model '" << formula.getRewardModelName() << "' does not exist.");
            return formula.getRewardModelName();
        }

        STORM_LOG_THROW(model.hasRewardModel(), storm::exceptions::InvalidPropertyException,
                        "Distributional model checking requires a reward model, but the model has none.");
        STORM_LOG_THROW(model.hasUniqueRewardModel(), storm::exceptions::InvalidPropertyException,
                        "Distributional model checking requires an explicit reward model name if the model has multiple reward models.");
        return model.getUniqueRewardModelName();
    }
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

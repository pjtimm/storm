#include "storm/modelchecker/distributional/DistributionalModelChecking.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/CheckTask.h"
#include "storm/modelchecker/distributional/DistributionalReachabilityPreprocessor.h"
#include "storm/modelchecker/distributional/DistributionalRewardReachabilityQuery.h"
#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/SparseMdpDistributionalValueIterationHelper.h"
#include "storm/modelchecker/results/CheckResult.h"
#include "storm/modelchecker/results/ExplicitDistributionalCheckResult.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/settings/SettingsManager.h"
#include "storm/settings/modules/DistributionalSettings.h"
#include "storm/solver/OptimizationDirection.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename SparseModelType, typename SolutionType>
std::unique_ptr<CheckResult> performDistributionalModelChecking(Environment const& env, SparseModelType const& model,
                                                                CheckTask<storm::logic::DistributionalFormula, SolutionType> const& checkTask) {
    STORM_LOG_THROW(checkTask.isOptimizationDirectionSet(), storm::exceptions::InvalidPropertyException,
                    "Distributional value iteration currently requires an explicit optimization direction.");
    STORM_LOG_THROW(storm::solver::minimize(checkTask.getOptimizationDirection()), storm::exceptions::NotSupportedException,
                    "Distributional value iteration currently supports only minimization objectives.");
    STORM_LOG_THROW(!checkTask.isProduceSchedulersSet(), storm::exceptions::NotSupportedException,
                    "Distributional value iteration does not support scheduler production yet.");

    auto query = parseDistributionalRewardReachabilityQuery(checkTask.getFormula());
    auto preprocessorResult = DistributionalReachabilityPreprocessor<SparseModelType>::preprocess(env, model, query, checkTask.isProduceSchedulersSet());
    auto const& settings = storm::settings::getModule<storm::settings::modules::DistributionalSettings>();
    auto options = DistributionalValueIterationOptions::fromSettings(settings);

    SparseMdpDistributionalValueIterationHelper<typename SparseModelType::ValueType> helper(
        preprocessorResult.targetAbsorbingTransitionMatrix, preprocessorResult.stateActionRewards, preprocessorResult.targetStates,
        preprocessorResult.properStates, options);
    auto result = helper.computeExpectedRewardOptimalDistributions();
    return std::make_unique<ExplicitDistributionalCheckResult<SolutionType>>(std::move(result.distributions), std::move(result.finiteDistributionStates));
}

template std::unique_ptr<CheckResult> performDistributionalModelChecking<storm::models::sparse::Mdp<double>, double>(
    Environment const& env, storm::models::sparse::Mdp<double> const& model, CheckTask<storm::logic::DistributionalFormula, double> const& checkTask);

template std::unique_ptr<CheckResult> performDistributionalModelChecking<storm::models::sparse::Mdp<storm::RationalNumber>, storm::RationalNumber>(
    Environment const& env, storm::models::sparse::Mdp<storm::RationalNumber> const& model,
    CheckTask<storm::logic::DistributionalFormula, storm::RationalNumber> const& checkTask);

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

#include "storm/modelchecker/distributional/DistributionalModelChecking.h"

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/NotImplementedException.h"
#include "storm/modelchecker/distributional/DistributionalReachabilityPreprocessor.h"
#include "storm/modelchecker/results/CheckResult.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/utility/Stopwatch.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<typename SparseModelType>
std::unique_ptr<CheckResult> performDistributionalModelChecking(Environment const& env, SparseModelType const& model,
                                                                storm::logic::DistributionalFormula const& formula, bool produceScheduler) {
    auto preprocessorResult = DistributionalReachabilityPreprocessor<SparseModelType>::preprocess(env, model, formula, produceScheduler);

    STORM_LOG_THROW(false, storm::exceptions::NotImplementedException,
                    "Distributional value iteration is not implemented yet for reward model '" << preprocessorResult.rewardModelName << "'.");
}

template std::unique_ptr<CheckResult> performDistributionalModelChecking<storm::models::sparse::Mdp<double>>(Environment const& env,
                                                                                                             storm::models::sparse::Mdp<double> const& model,
                                                                                                             storm::logic::DistributionalFormula const& formula,
                                                                                                             bool produceScheduler);

template std::unique_ptr<CheckResult> performDistributionalModelChecking<storm::models::sparse::Mdp<storm::RationalNumber>>(
    Environment const& env, storm::models::sparse::Mdp<storm::RationalNumber> const& model, storm::logic::DistributionalFormula const& formula,
    bool produceScheduler);

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

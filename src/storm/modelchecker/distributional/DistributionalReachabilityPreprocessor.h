#pragma once

#include <memory>
#include <string>

#include "storm/exceptions/InvalidPropertyException.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/logic/Formulas.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

template<class SparseMdpModelType>
struct DistributionalReachabilityPreprocessorResult {
    using RewardModelType = typename SparseMdpModelType::RewardModelType;

    std::string rewardModelName;
    RewardModelType const* rewardModel;
    std::shared_ptr<storm::logic::Formula const> targetFormula;
};

template<class SparseMdpModelType>
class DistributionalReachabilityPreprocessor {
   public:
    using Result = DistributionalReachabilityPreprocessorResult<SparseMdpModelType>;

    static Result preprocess(SparseMdpModelType const& model, storm::logic::DistributionalFormula const& formula) {
        storm::logic::Formula const& subformula = formula.getSubformula();
        STORM_LOG_THROW(subformula.isRewardOperatorFormula(), storm::exceptions::InvalidPropertyException,
                        "Distributional model checking currently requires a reward operator formula, but got '" << subformula << "'.");

        storm::logic::RewardOperatorFormula const& rewardOperatorFormula = subformula.asRewardOperatorFormula();
        STORM_LOG_THROW(rewardOperatorFormula.hasQuantitativeResult(), storm::exceptions::InvalidPropertyException,
                        "Distributional model checking currently requires a quantitative reward query without a comparison bound.");
        STORM_LOG_THROW(
            rewardOperatorFormula.getSubformula().isReachabilityRewardFormula(), storm::exceptions::InvalidPropertyException,
            "Distributional model checking currently requires a reachability reward formula, but got '" << rewardOperatorFormula.getSubformula() << "'.");

        storm::logic::EventuallyFormula const& reachabilityRewardFormula = rewardOperatorFormula.getSubformula().asReachabilityRewardFormula();
        STORM_LOG_THROW(reachabilityRewardFormula.getSubformula().isStateFormula(), storm::exceptions::InvalidPropertyException,
                        "Distributional model checking currently requires a state target formula.");

        std::string rewardModelName = getRewardModelName(model, rewardOperatorFormula);
        auto const& rewardModel = model.getRewardModel(rewardModelName);
        STORM_LOG_THROW(!rewardModel.hasTransitionRewards(), storm::exceptions::NotSupportedException,
                        "Distributional model checking does not support transition rewards yet.");

        return Result{rewardModelName, &rewardModel, reachabilityRewardFormula.getSubformula().asSharedPointer()};
    }

   private:
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

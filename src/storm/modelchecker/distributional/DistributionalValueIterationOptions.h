#pragma once

#include <cstdint>

#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/distributional/DistributionalCvarInterpretation.h"
#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/modelchecker/distributional/RewardDistributionRepresentation.h"
#include "storm/settings/modules/DistributionalSettings.h"
#include "storm/solver/OptimizationDirection.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

struct DistributionalValueIterationOptions {
    using Representation = RewardDistributionRepresentation;
    enum class Objective { RiskNeutral, Cvar };

    Representation representation = Representation::Categorical;
    uint64_t atoms = 101;
    uint64_t stepSize = 1;
    double precision = 1e-6;
    uint64_t maximalIterations = 10000;
    Objective objective = Objective::RiskNeutral;
    double alpha = 0.05;
    uint64_t budgetAtoms = 101;
    storm::solver::OptimizationDirection optimizationDirection = storm::solver::OptimizationDirection::Minimize;
    DistributionalCvarInterpretation cvarInterpretation = DistributionalCvarInterpretation::Cost;

    RewardDistributionOptions toRewardDistributionOptions() const {
        return RewardDistributionOptions{representation, atoms, stepSize};
    }

    static DistributionalValueIterationOptions fromSettings(
        storm::settings::modules::DistributionalSettings const& settings,
        storm::solver::OptimizationDirection optimizationDirection = storm::solver::OptimizationDirection::Minimize) {
        DistributionalValueIterationOptions options;
        options.representation = convertRepresentation(settings.getRepresentation());
        options.atoms = settings.getNumberOfAtoms();
        options.stepSize = settings.getRewardStepSize();
        options.precision = settings.getPrecision();
        options.maximalIterations = settings.getMaximalIterationCount();
        options.objective = convertObjective(settings.getObjective());
        options.alpha = settings.getAlpha();
        options.budgetAtoms = settings.getNumberOfBudgetAtoms();
        options.optimizationDirection = optimizationDirection;
        options.cvarInterpretation = resolveDistributionalCvarInterpretation(settings.getCvarInterpretationSelection(), optimizationDirection);
        options.validate();
        return options;
    }

    void validate() const {
        STORM_LOG_THROW(atoms > 0, storm::exceptions::NotSupportedException, "Distributional value iteration requires a positive atom count.");
        STORM_LOG_THROW(stepSize > 0, storm::exceptions::NotSupportedException,
                        "Distributional value iteration requires a positive categorical reward step size.");
        STORM_LOG_THROW(representation != Representation::Quantile, storm::exceptions::NotSupportedException,
                        "Distributional value iteration does not support quantile reward distributions yet.");
        STORM_LOG_THROW(alpha > 0.0 && alpha < 1.0, storm::exceptions::NotSupportedException,
                        "Distributional CVaR requires alpha to be in the interval (0, 1).");
        STORM_LOG_THROW(budgetAtoms > 0, storm::exceptions::NotSupportedException, "Distributional CVaR requires a positive budget atom count.");
    }

   private:
    static Objective convertObjective(storm::settings::modules::DistributionalSettings::Objective objective) {
        using SettingsObjective = storm::settings::modules::DistributionalSettings::Objective;
        switch (objective) {
            case SettingsObjective::RiskNeutral:
                return Objective::RiskNeutral;
            case SettingsObjective::Cvar:
                return Objective::Cvar;
        }
        STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Unknown distributional objective.");
    }

    static Representation convertRepresentation(storm::settings::modules::DistributionalSettings::Representation representation) {
        using SettingsRepresentation = storm::settings::modules::DistributionalSettings::Representation;
        switch (representation) {
            case SettingsRepresentation::Categorical:
                return Representation::Categorical;
            case SettingsRepresentation::Quantile:
                return Representation::Quantile;
        }
        STORM_LOG_THROW(false, storm::exceptions::NotSupportedException, "Unknown distributional reward representation.");
    }
};

}  // namespace distributional
}  // namespace modelchecker
}  // namespace storm

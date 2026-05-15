#pragma once

#include <cstdint>

#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/distributional/RewardDistribution.h"
#include "storm/modelchecker/distributional/RewardDistributionRepresentation.h"
#include "storm/settings/modules/DistributionalSettings.h"
#include "storm/utility/macros.h"

namespace storm {
namespace modelchecker {
namespace distributional {

struct DistributionalValueIterationOptions {
    using Representation = RewardDistributionRepresentation;

    Representation representation;
    uint64_t atoms;
    uint64_t stepSize;
    double precision;
    uint64_t maximalIterations;

    RewardDistributionOptions toRewardDistributionOptions() const {
        return RewardDistributionOptions{representation, atoms, stepSize};
    }

    static DistributionalValueIterationOptions fromSettings(storm::settings::modules::DistributionalSettings const& settings) {
        DistributionalValueIterationOptions options{convertRepresentation(settings.getRepresentation()), settings.getNumberOfAtoms(),
                                                    settings.getRewardStepSize(), settings.getPrecision(), settings.getMaximalIterationCount()};
        options.validate();
        return options;
    }

    void validate() const {
        STORM_LOG_THROW(atoms > 0, storm::exceptions::NotSupportedException,
                        "Distributional value iteration requires a positive atom count.");
        STORM_LOG_THROW(stepSize > 0, storm::exceptions::NotSupportedException,
                        "Distributional value iteration requires a positive categorical reward step size.");
        STORM_LOG_THROW(representation != Representation::Quantile, storm::exceptions::NotSupportedException,
                        "Distributional value iteration does not support quantile reward distributions yet.");
    }

   private:
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

#pragma once

#include <cstdint>
#include <string>

#include "storm/modelchecker/distributional/DistributionalCvarInterpretation.h"
#include "storm/settings/modules/ModuleSettings.h"

namespace storm {
namespace settings {
namespace modules {

class DistributionalSettings : public ModuleSettings {
   public:
    enum class Objective { RiskNeutral, Cvar };
    enum class Representation { Categorical, Quantile };

    DistributionalSettings();

    Objective getObjective() const;
    Representation getRepresentation() const;
    uint64_t getNumberOfAtoms() const;
    uint64_t getRewardStepSize() const;
    double getPrecision() const;
    uint64_t getMaximalIterationCount() const;
    uint64_t getNumberOfBudgetAtoms() const;
    double getAlpha() const;
    storm::modelchecker::distributional::DistributionalCvarInterpretationSelection getCvarInterpretationSelection() const;
    bool check() const override;

    static const std::string moduleName;

   private:
    static const std::string objectiveOptionName;
    static const std::string representationOptionName;
    static const std::string atomsOptionName;
    static const std::string stepSizeOptionName;
    static const std::string precisionOptionName;
    static const std::string maxIterationsOptionName;
    static const std::string budgetAtomsOptionName;
    static const std::string alphaOptionName;
    static const std::string interpretationOptionName;
};

}  // namespace modules
}  // namespace settings
}  // namespace storm

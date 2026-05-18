#include "storm/settings/modules/DistributionalSettings.h"

#include <vector>

#include "storm/exceptions/IllegalArgumentValueException.h"
#include "storm/settings/ArgumentBuilder.h"
#include "storm/settings/ArgumentValidators.h"
#include "storm/settings/OptionBuilder.h"
#include "storm/utility/macros.h"

namespace storm {
namespace settings {
namespace modules {

std::string const DistributionalSettings::moduleName = "distributional";
std::string const DistributionalSettings::representationOptionName = "representation";
std::string const DistributionalSettings::atomsOptionName = "atoms";
std::string const DistributionalSettings::stepSizeOptionName = "stepsize";
std::string const DistributionalSettings::precisionOptionName = "precision";
std::string const DistributionalSettings::maxIterationsOptionName = "maxiter";
std::string const DistributionalSettings::budgetAtomsOptionName = "budgetatoms";

DistributionalSettings::DistributionalSettings() : ModuleSettings(moduleName) {
    std::vector<std::string> representations = {"categorical", "quantile"};
    this->addOption(storm::settings::OptionBuilder(moduleName, representationOptionName, true,
                                                   "The finite projected distribution representation used for distributional model checking.")
                        .setIsAdvanced()
                        .addArgument(storm::settings::ArgumentBuilder::createStringArgument("name", "The representation to use.")
                                         .addValidatorString(ArgumentValidatorFactory::createMultipleChoiceValidator(representations))
                                         .setDefaultValueString("categorical")
                                         .build())
                        .build());
    this->addOption(storm::settings::OptionBuilder(moduleName, atomsOptionName, true, "The number of atoms used for finite reward distributions.")
                        .setIsAdvanced()
                        .addArgument(storm::settings::ArgumentBuilder::createUnsignedIntegerArgument("count", "The number of atoms.")
                                         .addValidatorUnsignedInteger(ArgumentValidatorFactory::createUnsignedGreaterValidator(0))
                                         .setDefaultValueUnsignedInteger(101)
                                         .build())
                        .build());
    this->addOption(
        storm::settings::OptionBuilder(moduleName, stepSizeOptionName, true,
                                       "The reward distance between neighboring categorical atoms. The categorical grid is [0, (atoms - 1) * stepsize], "
                                       "with the final atom collecting rewards beyond this upper bound.")
            .setIsAdvanced()
            .addArgument(storm::settings::ArgumentBuilder::createUnsignedIntegerArgument("value", "The categorical reward step size.")
                             .addValidatorUnsignedInteger(ArgumentValidatorFactory::createUnsignedGreaterValidator(0))
                             .setDefaultValueUnsignedInteger(1)
                             .build())
            .build());
    this->addOption(storm::settings::OptionBuilder(moduleName, precisionOptionName, true, "The convergence precision used for distributional value iteration.")
                        .setIsAdvanced()
                        .addArgument(storm::settings::ArgumentBuilder::createDoubleArgument("value", "The convergence precision.")
                                         .addValidatorDouble(ArgumentValidatorFactory::createDoubleGreaterValidator(0.0))
                                         .setDefaultValueDouble(1e-6)
                                         .build())
                        .build());
    this->addOption(storm::settings::OptionBuilder(moduleName, maxIterationsOptionName, true, "The maximal number of distributional value-iteration steps.")
                        .setIsAdvanced()
                        .addArgument(storm::settings::ArgumentBuilder::createUnsignedIntegerArgument("count", "The maximal number of iterations.")
                                         .addValidatorUnsignedInteger(ArgumentValidatorFactory::createUnsignedGreaterValidator(0))
                                         .setDefaultValueUnsignedInteger(10000)
                                         .build())
                        .build());
    this->addOption(
        storm::settings::OptionBuilder(moduleName, budgetAtomsOptionName, true, "The number of budget atoms used for risk-sensitive distributional objectives.")
            .setIsAdvanced()
            .addArgument(storm::settings::ArgumentBuilder::createUnsignedIntegerArgument("count", "The number of budget atoms.")
                             .addValidatorUnsignedInteger(ArgumentValidatorFactory::createUnsignedGreaterValidator(0))
                             .setDefaultValueUnsignedInteger(101)
                             .build())
            .build());
}

DistributionalSettings::Representation DistributionalSettings::getRepresentation() const {
    std::string representation = this->getOption(representationOptionName).getArgumentByName("name").getValueAsString();
    if (representation == "categorical") {
        return Representation::Categorical;
    } else if (representation == "quantile") {
        return Representation::Quantile;
    }
    STORM_LOG_THROW(false, storm::exceptions::IllegalArgumentValueException, "Unknown distributional representation '" << representation << "'.");
}

uint64_t DistributionalSettings::getNumberOfAtoms() const {
    return this->getOption(atomsOptionName).getArgumentByName("count").getValueAsUnsignedInteger();
}

uint64_t DistributionalSettings::getRewardStepSize() const {
    return this->getOption(stepSizeOptionName).getArgumentByName("value").getValueAsUnsignedInteger();
}

double DistributionalSettings::getPrecision() const {
    return this->getOption(precisionOptionName).getArgumentByName("value").getValueAsDouble();
}

uint64_t DistributionalSettings::getMaximalIterationCount() const {
    return this->getOption(maxIterationsOptionName).getArgumentByName("count").getValueAsUnsignedInteger();
}

uint64_t DistributionalSettings::getNumberOfBudgetAtoms() const {
    return this->getOption(budgetAtomsOptionName).getArgumentByName("count").getValueAsUnsignedInteger();
}

bool DistributionalSettings::check() const {
    return true;
}

}  // namespace modules
}  // namespace settings
}  // namespace storm

#include "storm-config.h"
#include "test/storm_gtest.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "storm-parsers/api/properties.h"
#include "storm-parsers/parser/FormulaParser.h"
#include "storm-parsers/parser/PrismParser.h"
#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/api/builder.h"
#include "storm/api/properties.h"
#include "storm/environment/Environment.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/logic/DistributionalFormula.h"
#include "storm/modelchecker/CheckTask.h"
#include "storm/modelchecker/distributional/DistributionalReachabilityPreprocessor.h"
#include "storm/modelchecker/distributional/DistributionalRewardReachabilityQuery.h"
#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h"
#include "storm/modelchecker/results/ExplicitDistributionalCheckResult.h"
#include "storm/models/sparse/Mdp.h"
#include "storm/models/sparse/StandardRewardModel.h"
#include "storm/models/sparse/StateLabeling.h"
#include "storm/settings/SettingMemento.h"
#include "storm/settings/SettingsManager.h"
#include "storm/settings/modules/DistributionalSettings.h"
#include "storm/solver/OptimizationDirection.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace {

using Mdp = storm::models::sparse::Mdp<double>;
using RewardModel = storm::models::sparse::StandardRewardModel<double>;

class DistributionalSettingsScope {
   public:
    DistributionalSettingsScope() {
        auto& settings = distributionalSettings();
        settings.restoreDefaults();
        mementos.push_back(settings.overrideOption("objective", false));
        mementos.push_back(settings.overrideOption("atoms", false));
        mementos.push_back(settings.overrideOption("stepsize", false));
        mementos.push_back(settings.overrideOption("budgetatoms", false));
        mementos.push_back(settings.overrideOption("alpha", false));
        mementos.push_back(settings.overrideOption("interpretation", false));
    }

    ~DistributionalSettingsScope() {
        distributionalSettings().restoreDefaults();
    }

    void apply(std::string const& settingsString) {
        storm::settings::mutableManager().setFromString(settingsString);
    }

   private:
    static storm::settings::modules::DistributionalSettings& distributionalSettings() {
        return dynamic_cast<storm::settings::modules::DistributionalSettings&>(
            storm::settings::mutableManager().getModule(storm::settings::modules::DistributionalSettings::moduleName));
    }

    std::vector<std::unique_ptr<storm::settings::SettingMemento>> mementos;
};

storm::storage::SparseMatrix<double> buildTwoStateTargetMatrix() {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 0, 1.0);
    return builder.build();
}

storm::storage::SparseMatrix<double> buildTwoStateTransitionRewardMatrix() {
    return buildTwoStateTargetMatrix();
}

storm::models::sparse::StateLabeling makeLabeling(uint64_t stateCount, std::vector<uint64_t> const& initialStates, std::vector<uint64_t> const& targetStates) {
    storm::models::sparse::StateLabeling labeling(stateCount);
    labeling.addLabel("init");
    for (auto state : initialStates) {
        labeling.addLabelToState("init", state);
    }
    labeling.addLabel("target");
    for (auto state : targetStates) {
        labeling.addLabelToState("target", state);
    }
    return labeling;
}

Mdp makeMdp(storm::storage::SparseMatrix<double>&& transitionMatrix, storm::models::sparse::StateLabeling&& labeling, RewardModel&& rewardModel) {
    std::unordered_map<std::string, RewardModel> rewardModels;
    rewardModels.emplace("", std::move(rewardModel));
    return Mdp(std::move(transitionMatrix), std::move(labeling), std::move(rewardModels));
}

Mdp makeTwoStateTargetMdp(std::vector<double> stateRewards, std::vector<double> stateActionRewards, std::vector<uint64_t> initialStates = {0},
                          std::optional<storm::storage::SparseMatrix<double>> transitionRewards = std::nullopt) {
    RewardModel rewardModel(std::optional<std::vector<double>>(std::move(stateRewards)), std::optional<std::vector<double>>(std::move(stateActionRewards)),
                            std::move(transitionRewards));
    return makeMdp(buildTwoStateTargetMatrix(), makeLabeling(2, initialStates, {1}), std::move(rewardModel));
}

Mdp makeImproperInitialMdp() {
    storm::storage::SparseMatrixBuilder<double> builder(2, 2, 2, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 0, 1.0);
    builder.newRowGroup(1);
    builder.addNextValue(1, 1, 1.0);
    RewardModel rewardModel(std::optional<std::vector<double>>(std::vector<double>{0.0, 0.0}),
                            std::optional<std::vector<double>>(std::vector<double>{0.0, 0.0}));
    return makeMdp(builder.build(), makeLabeling(2, {0}, {1}), std::move(rewardModel));
}

std::shared_ptr<storm::logic::DistributionalFormula const> makeDistributionalFormula(std::string const& formulaText) {
    storm::parser::FormulaParser formulaParser;
    return std::make_shared<storm::logic::DistributionalFormula>(formulaParser.parseSingleFormulaFromString(formulaText));
}

storm::modelchecker::distributional::DistributionalReachabilityPreprocessor<Mdp>::Result preprocess(Mdp const& mdp,
                                                                                                    std::string const& formulaText = "Rmin=? [F \"target\"]") {
    storm::Environment env;
    auto formula = makeDistributionalFormula(formulaText);
    auto query = storm::modelchecker::distributional::parseDistributionalRewardReachabilityQuery(*formula);
    return storm::modelchecker::distributional::DistributionalReachabilityPreprocessor<Mdp>::preprocess(env, mdp, query, false);
}

std::string distributionalChoiceModelString() {
    return R"(mdp

module distributional_choice
    s : [0..6] init 0;

    [fast] s=0 -> 0.85 : (s'=1) + 0.15 : (s'=2);
    [safe] s=0 -> (s'=3);

    [fast_low] s=1 -> (s'=6);
    [fast_high] s=2 -> (s'=6);

    [safe_first] s=3 -> (s'=4);
    [safe_second] s=4 -> 0.5 : (s'=5) + 0.5 : (s'=6);
    [safe_tail] s=5 -> (s'=6);

    [done] s=6 -> (s'=6);
endmodule

rewards "cost"
    [fast] true : 0;
    [safe] true : 0;
    [fast_low] true : 2;
    [fast_high] true : 30;
    [safe_first] true : 3;
    [safe_second] true : 3;
    [safe_tail] true : 1;
    [done] true : 0;
endrewards

label "target" = s=6;
)";
}

std::unique_ptr<storm::modelchecker::CheckResult> checkDistributionalFromStrings(std::string const& programString, std::string const& formulaString,
                                                                                 std::string const& settingsString) {
    DistributionalSettingsScope settings;
    settings.apply(settingsString);

    auto program = storm::parser::PrismParser::parseFromString(programString, "distributional-test.nm");
    auto formulas = storm::api::extractFormulasFromProperties(storm::api::parsePropertiesForPrismProgram(formulaString, program));
    STORM_LOG_THROW(formulas.size() == 1, storm::exceptions::NotSupportedException, "Expected exactly one formula in distributional model-checker test.");
    auto mdp = storm::api::buildSparseModel<double>(program, formulas)->template as<Mdp>();
    storm::modelchecker::SparseMdpPrctlModelChecker<Mdp> checker(*mdp);
    storm::Environment env;
    auto distributionalFormula = std::make_shared<storm::logic::DistributionalFormula>(formulas.front());
    return checker.check(env, *distributionalFormula);
}

void expectInitialDistribution(std::unique_ptr<storm::modelchecker::CheckResult> const& result, double expectedValue,
                               std::string const& expectedDistributionText) {
    ASSERT_TRUE(result->isExplicitDistributionalCheckResult());
    auto const& distributionalResult = result->asExplicitDistributionalCheckResult<double>();
    EXPECT_FALSE(distributionalResult.isResultForAllStates());
    ASSERT_TRUE(distributionalResult.hasFiniteDistribution(0));
    EXPECT_NEAR(expectedValue, distributionalResult.getExpectedValue(0), 1e-8);

    std::stringstream stream;
    stream << *result;
    EXPECT_NE(std::string::npos, stream.str().find(expectedDistributionText));
}

void expectParsedInterpretation(std::string const& settingsString,
                                storm::modelchecker::distributional::DistributionalCvarInterpretationSelection expectedInterpretation) {
    DistributionalSettingsScope settingsScope;
    auto& settings = dynamic_cast<storm::settings::modules::DistributionalSettings&>(
        storm::settings::mutableManager().getModule(storm::settings::modules::DistributionalSettings::moduleName));

    settingsScope.apply(settingsString);
    EXPECT_EQ(expectedInterpretation, settings.getCvarInterpretationSelection());
}

}  // namespace

TEST(SparseMdpDistributionalPreprocessorTest, NormalizesRewardsAndMakesTargetsAbsorbing) {
    auto mdp = makeTwoStateTargetMdp({10.0, 20.0}, {1.0, 2.0, 3.0});

    auto result = preprocess(mdp);

    ASSERT_EQ(3ul, result.stateActionRewards.size());
    EXPECT_DOUBLE_EQ(11.0, result.stateActionRewards[0]);
    EXPECT_DOUBLE_EQ(12.0, result.stateActionRewards[1]);
    EXPECT_DOUBLE_EQ(0.0, result.stateActionRewards[2]);
    EXPECT_TRUE(result.targetStates.get(1));
    EXPECT_TRUE(result.properStates.get(0));
    EXPECT_TRUE(result.properStates.get(1));

    auto targetRow = result.targetAbsorbingTransitionMatrix.getRow(2);
    ASSERT_EQ(1ul, targetRow.getNumberOfEntries());
    auto entry = *targetRow.begin();
    EXPECT_EQ(1ul, entry.getColumn());
    EXPECT_DOUBLE_EQ(1.0, entry.getValue());
}

TEST(SparseMdpDistributionalPreprocessorTest, RejectsTransitionRewards) {
    auto mdp =
        makeTwoStateTargetMdp({0.0, 0.0}, {1.0, 0.0, 0.0}, {0}, std::optional<storm::storage::SparseMatrix<double>>(buildTwoStateTransitionRewardMatrix()));

    STORM_SILENT_EXPECT_THROW(preprocess(mdp), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalPreprocessorTest, RejectsNegativeAndNonIntegerRewards) {
    auto negativeRewardMdp = makeTwoStateTargetMdp({0.0, 0.0}, {-1.0, 0.0, 0.0});
    STORM_SILENT_EXPECT_THROW(preprocess(negativeRewardMdp), storm::exceptions::NotSupportedException);

    auto nonIntegerRewardMdp = makeTwoStateTargetMdp({0.0, 0.0}, {0.5, 0.0, 0.0});
    STORM_SILENT_EXPECT_THROW(preprocess(nonIntegerRewardMdp), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalPreprocessorTest, RejectsMultipleInitialStatesAndImproperInitialState) {
    auto multipleInitialStatesMdp = makeTwoStateTargetMdp({0.0, 0.0}, {0.0, 0.0, 0.0}, {0, 1});
    STORM_SILENT_EXPECT_THROW(preprocess(multipleInitialStatesMdp), storm::exceptions::NotSupportedException);

    auto improperInitialStateMdp = makeImproperInitialMdp();
    STORM_SILENT_EXPECT_THROW(preprocess(improperInitialStateMdp), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalModelCheckingTest, ReturnsDistributionalResultThroughSparseMdpPrctlPath) {
    auto mdp = makeTwoStateTargetMdp({0.0, 0.0}, {3.0, 5.0, 0.0});
    storm::modelchecker::SparseMdpPrctlModelChecker<Mdp> checker(mdp);
    storm::Environment env;
    auto formula = makeDistributionalFormula("Rmin=? [F \"target\"]");

    auto result = checker.check(env, *formula);

    ASSERT_TRUE(result->isExplicitDistributionalCheckResult());
    EXPECT_FALSE(result->isQuantitative());
    auto const& distributionalResult = result->asExplicitDistributionalCheckResult<double>();
    ASSERT_TRUE(distributionalResult.hasFiniteDistribution(0));
    EXPECT_DOUBLE_EQ(3.0, distributionalResult.getExpectedValue(0));
}

TEST(SparseMdpDistributionalModelCheckingTest, RejectsUnsupportedOptimizationAndSchedulerProduction) {
    auto mdp = makeTwoStateTargetMdp({0.0, 0.0}, {3.0, 5.0, 0.0});
    storm::modelchecker::SparseMdpPrctlModelChecker<Mdp> checker(mdp);
    storm::Environment env;

    auto maxFormula = makeDistributionalFormula("Rmax=? [F \"target\"]");
    STORM_SILENT_EXPECT_THROW(checker.check(env, *maxFormula), storm::exceptions::NotSupportedException);

    auto schedulerFormula = makeDistributionalFormula("Rmin=? [F \"target\"]");
    storm::modelchecker::CheckTask<storm::logic::Formula, double> schedulerTask(*schedulerFormula);
    schedulerTask.setProduceSchedulers();
    STORM_SILENT_EXPECT_THROW(checker.check(env, schedulerTask), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalSettingsTest, ParsesCvarInterpretationSelection) {
    expectParsedInterpretation("--distributional:interpretation auto", storm::modelchecker::distributional::DistributionalCvarInterpretationSelection::Auto);
    expectParsedInterpretation("--distributional:interpretation cost", storm::modelchecker::distributional::DistributionalCvarInterpretationSelection::Cost);
    expectParsedInterpretation("--distributional:interpretation reward",
                               storm::modelchecker::distributional::DistributionalCvarInterpretationSelection::Reward);
}

TEST(SparseMdpDistributionalSettingsTest, ResolvesAutoCvarInterpretationFromOptimizationDirection) {
    DistributionalSettingsScope settingsScope;
    auto& settings = dynamic_cast<storm::settings::modules::DistributionalSettings&>(
        storm::settings::mutableManager().getModule(storm::settings::modules::DistributionalSettings::moduleName));

    settingsScope.apply("--distributional:objective cvar --distributional:interpretation auto");
    auto minOptions =
        storm::modelchecker::distributional::DistributionalValueIterationOptions::fromSettings(settings, storm::solver::OptimizationDirection::Minimize);
    EXPECT_EQ(storm::modelchecker::distributional::DistributionalCvarInterpretation::Cost, minOptions.cvarInterpretation);

    auto maxOptions =
        storm::modelchecker::distributional::DistributionalValueIterationOptions::fromSettings(settings, storm::solver::OptimizationDirection::Maximize);
    EXPECT_EQ(storm::modelchecker::distributional::DistributionalCvarInterpretation::Reward, maxOptions.cvarInterpretation);
}

TEST(SparseMdpDistributionalModelCheckingTest, ComputesRiskNeutralResultFromPrismStrings) {
    auto result = checkDistributionalFromStrings(distributionalChoiceModelString(), "R{\"cost\"}min=? [ F \"target\" ];",
                                                 "--distributional:objective risk-neutral --distributional:atoms 41 --distributional:stepsize 1");

    ASSERT_TRUE(result->isExplicitDistributionalCheckResult());
    auto const& distributionalResult = result->asExplicitDistributionalCheckResult<double>();
    ASSERT_TRUE(distributionalResult.hasFiniteDistribution(0));
    EXPECT_NEAR(6.2, distributionalResult.getExpectedValue(0), 1e-8);

    std::stringstream stream;
    stream << *result;
    EXPECT_NE(std::string::npos, stream.str().find("{2: 0.85, 30: 0.15}"));
}

TEST(SparseMdpDistributionalModelCheckingTest, ComputesCvarResultFromPrismStrings) {
    auto result = checkDistributionalFromStrings(distributionalChoiceModelString(), "R{\"cost\"}min=? [ F \"target\" ];",
                                                 "--distributional:objective cvar --distributional:alpha 0.25 --distributional:atoms 41 "
                                                 "--distributional:stepsize 1");

    ASSERT_TRUE(result->isExplicitDistributionalCheckResult());
    auto const& distributionalResult = result->asExplicitDistributionalCheckResult<double>();
    EXPECT_FALSE(distributionalResult.isResultForAllStates());
    ASSERT_TRUE(distributionalResult.hasFiniteDistribution(0));
    EXPECT_NEAR(6.5, distributionalResult.getExpectedValue(0), 1e-8);

    std::stringstream stream;
    stream << *result;
    EXPECT_NE(std::string::npos, stream.str().find("{6: 0.5, 7: 0.5}"));
}

TEST(SparseMdpDistributionalModelCheckingTest, CvarIgnoresBudgetAtomsFromPrismStrings) {
    std::string const cvarSettings = "--distributional:objective cvar --distributional:alpha 0.25 --distributional:atoms 41 --distributional:stepsize 1";

    auto oneBudgetAtom = checkDistributionalFromStrings(distributionalChoiceModelString(), "R{\"cost\"}min=? [ F \"target\" ];",
                                                        cvarSettings + " --distributional:budgetatoms 1");
    expectInitialDistribution(oneBudgetAtom, 6.5, "{6: 0.5, 7: 0.5}");

    auto severalBudgetAtoms = checkDistributionalFromStrings(distributionalChoiceModelString(), "R{\"cost\"}min=? [ F \"target\" ];",
                                                             cvarSettings + " --distributional:budgetatoms 3");
    expectInitialDistribution(severalBudgetAtoms, 6.5, "{6: 0.5, 7: 0.5}");
}

TEST(SparseMdpDistributionalModelCheckingTest, ComputesCvarResultsForExplicitInterpretationsFromPrismStrings) {
    std::string const cvarSettings = "--distributional:objective cvar --distributional:alpha 0.25 --distributional:atoms 41 --distributional:stepsize 1";

    auto maxRewardAuto = checkDistributionalFromStrings(distributionalChoiceModelString(), "R{\"cost\"}max=? [ F \"target\" ];", cvarSettings);
    expectInitialDistribution(maxRewardAuto, 6.5, "{6: 0.5, 7: 0.5}");

    auto maxCost = checkDistributionalFromStrings(distributionalChoiceModelString(), "R{\"cost\"}max=? [ F \"target\" ];",
                                                  cvarSettings + " --distributional:interpretation cost");
    expectInitialDistribution(maxCost, 6.2, "{2: 0.85, 30: 0.15}");

    auto minReward = checkDistributionalFromStrings(distributionalChoiceModelString(), "R{\"cost\"}min=? [ F \"target\" ];",
                                                    cvarSettings + " --distributional:interpretation reward");
    expectInitialDistribution(minReward, 6.2, "{2: 0.85, 30: 0.15}");
}

TEST(SparseMdpDistributionalModelCheckingTest, RejectsCvarCyclicProperSubsystemFromPrismString) {
    std::string const program = R"(mdp
module cyclic
    s : [0..1] init 0;
    [] s=0 -> 0.5 : (s'=0) + 0.5 : (s'=1);
    [] s=1 -> (s'=1);
endmodule
rewards "cost"
    s=0 : 1;
    s=1 : 0;
endrewards
label "target" = s=1;
)";

    STORM_SILENT_EXPECT_THROW(checkDistributionalFromStrings(program, "R{\"cost\"}min=? [ F \"target\" ];",
                                                             "--distributional:objective cvar --distributional:alpha 0.25 "
                                                             "--distributional:atoms 10 --distributional:stepsize 1"),
                              storm::exceptions::NotSupportedException);
}

TEST(SparseMdpDistributionalModelCheckingTest, RejectsCvarModelsWithImproperOriginalStatesFromPrismString) {
    std::string const program = R"(mdp
module partially_improper
    s : [0..2] init 0;
    [] s=0 -> (s'=2);
    [] s=0 -> (s'=1);
    [] s=1 -> (s'=1);
    [] s=2 -> (s'=2);
endmodule
rewards "cost"
    true : 0;
endrewards
label "target" = s=2;
)";

    STORM_SILENT_EXPECT_THROW(checkDistributionalFromStrings(program, "R{\"cost\"}min=? [ F \"target\" ];",
                                                             "--distributional:objective cvar --distributional:alpha 0.25 "
                                                             "--distributional:atoms 10 --distributional:stepsize 1"),
                              storm::exceptions::NotSupportedException);
}

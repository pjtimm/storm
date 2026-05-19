#include "storm-config.h"
#include "test/storm_gtest.h"

#include <vector>

#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/SparseMdpCvarObjective.h"
#include "storm/modelchecker/distributional/SparseMdpCvarPreprocessor.h"
#include "storm/modelchecker/distributional/SparseMdpRiskNeutralObjective.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace {

storm::modelchecker::distributional::DistributionalValueIterationOptions makeRiskNeutralOptions(uint64_t atoms = 128, uint64_t stepSize = 1) {
    return storm::modelchecker::distributional::DistributionalValueIterationOptions{
        storm::modelchecker::distributional::RewardDistributionRepresentation::Categorical, atoms, stepSize, 1e-10, 10000};
}

storm::modelchecker::distributional::DistributionalValueIterationOptions makeCvarOptions(uint64_t atoms = 128, uint64_t stepSize = 1,
                                                                                        uint64_t budgetAtoms = 3) {
    auto options = makeRiskNeutralOptions(atoms, stepSize);
    options.objective = storm::modelchecker::distributional::DistributionalValueIterationOptions::Objective::Cvar;
    options.budgetAtoms = budgetAtoms;
    return options;
}

}  // namespace

TEST(SparseMdpCvarPreprocessorTest, ComputesBoundsForProperDag) {
    storm::storage::SparseMatrixBuilder<double> builder(5, 4, 5, true, true, 4);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 2, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 3, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 3, 1.0);
    builder.newRowGroup(4);
    builder.addNextValue(4, 3, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {2.0, 3.0, 4.0, 1.0, 0.0};
    storm::storage::BitVector targetStates(4, std::vector<uint64_t>{3});
    storm::storage::BitVector properStates(4, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, 3);
    auto result = preprocessor.computeRewardBounds();

    EXPECT_TRUE(result.finiteRewardStates.get(0));
    EXPECT_EQ(0ul, result.initialState);
    ASSERT_EQ(4ul, result.lowerRewardBounds.size());
    ASSERT_EQ(4ul, result.upperRewardBounds.size());
    EXPECT_DOUBLE_EQ(4.0, result.lowerRewardBounds[0]);
    EXPECT_DOUBLE_EQ(6.0, result.upperRewardBounds[0]);
    EXPECT_DOUBLE_EQ(4.0, result.lowerRewardBounds[1]);
    EXPECT_DOUBLE_EQ(4.0, result.upperRewardBounds[1]);
    EXPECT_DOUBLE_EQ(1.0, result.lowerRewardBounds[2]);
    EXPECT_DOUBLE_EQ(1.0, result.upperRewardBounds[2]);
    EXPECT_DOUBLE_EQ(0.0, result.lowerRewardBounds[3]);
    EXPECT_DOUBLE_EQ(0.0, result.upperRewardBounds[3]);
    EXPECT_DOUBLE_EQ(4.0, result.initialLowerRewardBound);
    EXPECT_DOUBLE_EQ(6.0, result.initialUpperRewardBound);
    ASSERT_EQ(3ul, result.getNumberOfBudgetAtoms());
    EXPECT_DOUBLE_EQ(4.0, result.getBudgetValue(0));
    EXPECT_DOUBLE_EQ(5.0, result.getBudgetValue(1));
    EXPECT_DOUBLE_EQ(6.0, result.getBudgetValue(2));
    EXPECT_EQ(0ul, result.getNextBudgetIndex(2, 2.0));
    EXPECT_EQ(1ul, result.getNextBudgetIndex(2, 1.0));
}

TEST(SparseMdpCvarPreprocessorTest, BuildsEvenlySpacedBudgetGrid) {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 10.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, 4);
    auto result = preprocessor.computeRewardBounds();

    ASSERT_EQ(4ul, result.getNumberOfBudgetAtoms());
    EXPECT_DOUBLE_EQ(0.0, result.getBudgetValue(0));
    EXPECT_NEAR(10.0 / 3.0, result.getBudgetValue(1), 1e-12);
    EXPECT_NEAR(20.0 / 3.0, result.getBudgetValue(2), 1e-12);
    EXPECT_DOUBLE_EQ(10.0, result.getBudgetValue(3));
    EXPECT_EQ(1ul, result.getNextBudgetIndex(3, 4.0));
    EXPECT_EQ(2ul, result.getNextBudgetIndex(2, 0.0));
    EXPECT_EQ(0ul, result.getNextBudgetIndex(1, 20.0));
}

TEST(SparseMdpCvarPreprocessorTest, AcceptsSingletonInitialSupportWithOneBudgetAtom) {
    storm::storage::SparseMatrixBuilder<double> builder(2, 2, 2, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.newRowGroup(1);
    builder.addNextValue(1, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {5.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, 1);
    auto result = preprocessor.computeRewardBounds();

    ASSERT_EQ(1ul, result.getNumberOfBudgetAtoms());
    EXPECT_DOUBLE_EQ(5.0, result.getBudgetValue(0));
    EXPECT_EQ(0ul, result.getNextBudgetIndex(0, 0.0));
    EXPECT_EQ(0ul, result.getNextBudgetIndex(0, 5.0));
}

TEST(SparseMdpCvarPreprocessorTest, RejectsOneBudgetAtomForNonSingletonInitialSupport) {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 10.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, 1);
    STORM_SILENT_EXPECT_THROW(preprocessor.computeRewardBounds(), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpCvarObjectiveTest, ProductIndexRoundTrips) {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 10.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);
    auto options = makeCvarOptions(128, 1, 4);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, options.budgetAtoms);
    auto preprocessorResult = preprocessor.computeRewardBounds();
    storm::modelchecker::distributional::SparseMdpCvarObjective<double> objective(matrix, rewards, targetStates, properStates, options, preprocessorResult);

    EXPECT_EQ(2ul, objective.getStateCount());
    EXPECT_EQ(4ul, objective.getBudgetCount());
    EXPECT_EQ(8ul, objective.getProductStateCount());
    for (uint64_t state = 0; state < objective.getStateCount(); ++state) {
        for (uint64_t budgetIndex = 0; budgetIndex < objective.getBudgetCount(); ++budgetIndex) {
            uint64_t const productState = objective.getProductStateIndex(state, budgetIndex);
            EXPECT_EQ(state, objective.getOriginalState(productState));
            EXPECT_EQ(budgetIndex, objective.getBudgetIndex(productState));
        }
    }
}

TEST(SparseMdpCvarObjectiveTest, MemoizesProductDistributionsLazily) {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 10.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);
    auto options = makeCvarOptions(5, 1, 4);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, options.budgetAtoms);
    auto preprocessorResult = preprocessor.computeRewardBounds();
    storm::modelchecker::distributional::SparseMdpCvarObjective<double> objective(matrix, rewards, targetStates, properStates, options, preprocessorResult);
    auto cache = objective.createProductDistributionCache();

    EXPECT_EQ(0ul, cache.getCachedDistributionCount());

    uint64_t const nonTargetProductState = objective.getProductStateIndex(0, 2);
    EXPECT_TRUE(objective.isFiniteProductState(nonTargetProductState));
    auto const& nonTargetDistribution = objective.getOrInitializeProductDistribution(cache, nonTargetProductState);
    EXPECT_TRUE(nonTargetDistribution.isCategorical());
    EXPECT_DOUBLE_EQ(4.0, nonTargetDistribution.getProjectedExpectedValue());
    EXPECT_TRUE(cache.hasDistribution(nonTargetProductState));
    EXPECT_EQ(1ul, cache.getCachedDistributionCount());

    auto const& sameDistribution = objective.getOrInitializeProductDistribution(cache, nonTargetProductState);
    EXPECT_DOUBLE_EQ(nonTargetDistribution.getProjectedExpectedValue(), sameDistribution.getProjectedExpectedValue());
    EXPECT_EQ(1ul, cache.getCachedDistributionCount());

    uint64_t const targetProductState = objective.getProductStateIndex(1, 3);
    EXPECT_TRUE(objective.isFiniteProductState(targetProductState));
    auto const& targetDistribution = objective.getOrInitializeProductDistribution(cache, targetProductState);
    EXPECT_TRUE(targetDistribution.isExact());
    EXPECT_DOUBLE_EQ(0.0, targetDistribution.getProjectedExpectedValue());
    EXPECT_TRUE(cache.hasDistribution(targetProductState));
    EXPECT_EQ(2ul, cache.getCachedDistributionCount());
}

TEST(SparseMdpCvarObjectiveTest, MasksImproperOriginalStatesWhenInitializedDirectly) {
    storm::storage::SparseMatrixBuilder<double> builder(2, 2, 2, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 0, 1.0);
    builder.newRowGroup(1);
    builder.addNextValue(1, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {0.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, std::vector<uint64_t>{1});
    auto options = makeCvarOptions(5, 1, 2);

    storm::modelchecker::distributional::SparseMdpCvarObjective<double>::PreprocessorResult preprocessorResult;
    preprocessorResult.lowerRewardBounds = {0.0, 0.0};
    preprocessorResult.upperRewardBounds = {0.0, 0.0};
    preprocessorResult.finiteRewardStates = properStates;
    preprocessorResult.initialState = 1;
    preprocessorResult.initialLowerRewardBound = 0.0;
    preprocessorResult.initialUpperRewardBound = 0.0;
    preprocessorResult.budgetGrid = {0.0, 1.0};

    storm::modelchecker::distributional::SparseMdpCvarObjective<double> objective(matrix, rewards, targetStates, properStates, options, preprocessorResult);
    auto cache = objective.createProductDistributionCache();

    for (uint64_t budgetIndex = 0; budgetIndex < objective.getBudgetCount(); ++budgetIndex) {
        uint64_t const improperProductState = objective.getProductStateIndex(0, budgetIndex);
        uint64_t const properProductState = objective.getProductStateIndex(1, budgetIndex);
        EXPECT_FALSE(objective.isFiniteProductState(improperProductState));
        EXPECT_TRUE(objective.isFiniteProductState(properProductState));
        STORM_SILENT_EXPECT_THROW(objective.getOrInitializeProductDistribution(cache, improperProductState), storm::exceptions::InvalidArgumentException);
        EXPECT_FALSE(cache.hasDistribution(improperProductState));
    }
}

TEST(SparseMdpCvarPreprocessorTest, RejectsProperNonTargetCycle) {
    storm::storage::SparseMatrixBuilder<double> builder(4, 3, 4, true, true, 3);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 2, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 0, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 2, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {1.0, 0.0, 1.0, 0.0};
    storm::storage::BitVector targetStates(3, std::vector<uint64_t>{2});
    storm::storage::BitVector properStates(3, true);

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, 3);
    STORM_SILENT_EXPECT_THROW(preprocessor.computeRewardBounds(), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpCvarPreprocessorTest, RejectsImproperStates) {
    storm::storage::SparseMatrixBuilder<double> builder(4, 3, 4, true, true, 3);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 2, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    builder.newRowGroup(3);
    builder.addNextValue(3, 2, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {4.0, 0.0, 0.0, 0.0};
    storm::storage::BitVector targetStates(3, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(3, std::vector<uint64_t>{0, 1});

    auto constructPreprocessor = [&]() {
        storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates, 0, 3);
        static_cast<void>(preprocessor);
    };
    STORM_SILENT_EXPECT_THROW(constructPreprocessor(), storm::exceptions::NotSupportedException);
}

TEST(SparseMdpCvarPreprocessorTest, RiskNeutralDviStillAcceptsProperCycle) {
    storm::storage::SparseMatrixBuilder<double> builder(2, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 0, 0.5);
    builder.addNextValue(0, 1, 0.5);
    builder.newRowGroup(1);
    builder.addNextValue(1, 1, 1.0);
    auto matrix = builder.build();

    std::vector<double> rewards = {1.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);

    storm::modelchecker::distributional::SparseMdpRiskNeutralObjective<double> objective(matrix, rewards, targetStates, properStates,
                                                                                        makeRiskNeutralOptions());
    auto result = objective.computeExpectedRewardOptimalDistributions();

    ASSERT_TRUE(result.finiteDistributionStates.get(0));
    EXPECT_NEAR(2.0, result.distributions[0].getProjectedExpectedValue(), 1e-5);
}

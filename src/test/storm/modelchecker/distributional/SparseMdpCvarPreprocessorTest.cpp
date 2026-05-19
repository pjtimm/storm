#include "storm-config.h"
#include "test/storm_gtest.h"

#include <vector>

#include "storm/exceptions/NotSupportedException.h"
#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/SparseMdpCvarPreprocessor.h"
#include "storm/modelchecker/distributional/SparseMdpRiskNeutralObjective.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace {

storm::modelchecker::distributional::DistributionalValueIterationOptions makeRiskNeutralOptions(uint64_t atoms = 128, uint64_t stepSize = 1) {
    return storm::modelchecker::distributional::DistributionalValueIterationOptions{
        storm::modelchecker::distributional::RewardDistributionRepresentation::Categorical, atoms, stepSize, 1e-10, 10000};
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

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates);
    auto result = preprocessor.computeRewardBounds();

    EXPECT_TRUE(result.finiteRewardStates.get(0));
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

    storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates);
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
        storm::modelchecker::distributional::SparseMdpCvarPreprocessor<double> preprocessor(matrix, rewards, targetStates, properStates);
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

#include "storm-config.h"
#include "test/storm_gtest.h"

#include <utility>
#include <vector>

#include "storm/adapters/RationalNumberAdapter.h"
#include "storm/exceptions/InvalidAccessException.h"
#include "storm/modelchecker/distributional/DistributionalValueIterationOptions.h"
#include "storm/modelchecker/distributional/SparseMdpDistributionalValueIterationHelper.h"
#include "storm/modelchecker/results/ExplicitDistributionalCheckResult.h"
#include "storm/modelchecker/results/ExplicitQualitativeCheckResult.h"
#include "storm/storage/BitVector.h"
#include "storm/storage/SparseMatrix.h"

namespace {

storm::modelchecker::distributional::DistributionalValueIterationOptions makeOptions(uint64_t atoms = 32, uint64_t stepSize = 1) {
    return storm::modelchecker::distributional::DistributionalValueIterationOptions{
        storm::modelchecker::distributional::RewardDistributionRepresentation::Categorical, atoms, stepSize, 1e-10, 10000};
}

}  // namespace

TEST(SparseMdpDistributionalValueIterationHelperTest, ReturnsDistributionForBestAcyclicChoice) {
    storm::storage::SparseMatrixBuilder<double> builder(3, 2, 3, true, true, 2);
    builder.newRowGroup(0);
    builder.addNextValue(0, 1, 1.0);
    builder.addNextValue(1, 1, 1.0);
    builder.newRowGroup(2);
    builder.addNextValue(2, 1, 1.0);
    auto matrix = builder.build();
    std::vector<double> rewards = {3.0, 5.0, 0.0};
    storm::storage::BitVector targetStates(2, std::vector<uint64_t>{1});
    storm::storage::BitVector properStates(2, true);

    storm::modelchecker::distributional::SparseMdpDistributionalValueIterationHelper<double> helper(matrix, rewards, targetStates, properStates, makeOptions());
    auto result = helper.computeExpectedRewardOptimalDistributions();

    ASSERT_EQ(2ul, result.distributions.size());
    ASSERT_TRUE(result.finiteDistributionStates.get(0));
    ASSERT_TRUE(result.distributions[0].isCategorical());
    auto const& masses = result.distributions[0].getCategoricalMasses();
    ASSERT_GT(masses.size(), 3ul);
    EXPECT_DOUBLE_EQ(1.0, masses[3]);
    EXPECT_DOUBLE_EQ(3.0, result.distributions[0].getExpectedValue());
    EXPECT_DOUBLE_EQ(0.0, result.distributions[1].getExpectedValue());
}

TEST(SparseMdpDistributionalValueIterationHelperTest, FiltersAvoidableImproperChoices) {
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

    storm::modelchecker::distributional::SparseMdpDistributionalValueIterationHelper<double> helper(matrix, rewards, targetStates, properStates, makeOptions());
    auto result = helper.computeExpectedRewardOptimalDistributions();

    EXPECT_TRUE(result.finiteDistributionStates.get(0));
    EXPECT_TRUE(result.finiteDistributionStates.get(1));
    EXPECT_FALSE(result.finiteDistributionStates.get(2));
    EXPECT_DOUBLE_EQ(4.0, result.distributions[0].getExpectedValue());

    auto const& masses = result.distributions[0].getCategoricalMasses();
    ASSERT_GT(masses.size(), 4ul);
    EXPECT_DOUBLE_EQ(1.0, masses[4]);
}

TEST(SparseMdpDistributionalValueIterationHelperTest, ConvergesOnProperCyclicModel) {
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

    storm::modelchecker::distributional::SparseMdpDistributionalValueIterationHelper<double> helper(matrix, rewards, targetStates, properStates,
                                                                                                    makeOptions(128));
    auto result = helper.computeExpectedRewardOptimalDistributions();

    ASSERT_TRUE(result.distributions[0].isCategorical());
    EXPECT_NEAR(2.0, result.distributions[0].getExpectedValue(), 1e-5);
    EXPECT_DOUBLE_EQ(0.0, result.distributions[1].getExpectedValue());
}

TEST(ExplicitDistributionalCheckResultTest, StoresAndFiltersFiniteDistributions) {
    using Distribution = storm::modelchecker::distributional::RewardDistribution<double>;

    std::vector<Distribution> distributions;
    distributions.push_back(Distribution::pointMass(2));
    distributions.push_back(Distribution::categoricalTail(storm::modelchecker::distributional::RewardDistributionOptions()));
    distributions.push_back(Distribution::pointMass(0));

    storm::storage::BitVector finiteStates(3, std::vector<uint64_t>{0, 2});

    storm::modelchecker::ExplicitDistributionalCheckResult<double> result(std::move(distributions), std::move(finiteStates));
    EXPECT_TRUE(result.isExplicit());
    EXPECT_TRUE(result.isResultForAllStates());
    EXPECT_TRUE(result.isExplicitDistributionalCheckResult());
    storm::modelchecker::CheckResult& checkResult = result;
    EXPECT_TRUE(checkResult.isExplicitDistributionalCheckResult());
    EXPECT_DOUBLE_EQ(2.0, checkResult.asExplicitDistributionalCheckResult<double>().getExpectedValue(0));
    EXPECT_TRUE(result.hasDistribution(0));
    EXPECT_FALSE(result.hasDistribution(1));
    EXPECT_TRUE(result.hasDistribution(2));
    EXPECT_DOUBLE_EQ(2.0, result.getExpectedValue(0));
    STORM_SILENT_EXPECT_THROW(result.getDistribution(1), storm::exceptions::InvalidAccessException);
    STORM_SILENT_EXPECT_THROW(result.getExpectedValue(1), storm::exceptions::InvalidAccessException);

    storm::storage::BitVector filterValues(3, std::vector<uint64_t>{1, 2});
    storm::modelchecker::ExplicitQualitativeCheckResult<double> filter(std::move(filterValues));
    result.filter(filter);

    EXPECT_FALSE(result.isResultForAllStates());
    EXPECT_FALSE(result.hasDistribution(1));
    EXPECT_TRUE(result.hasDistribution(2));
    STORM_SILENT_EXPECT_THROW(result.getExpectedValue(1), storm::exceptions::InvalidAccessException);
    EXPECT_DOUBLE_EQ(0.0, result.getExpectedValue(2));
}

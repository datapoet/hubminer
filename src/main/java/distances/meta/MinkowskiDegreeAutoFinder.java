/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 * 
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package distances.meta;

import data.neighbors.NeighborSetFinder;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import ioformat.SupervisedLoader;
import java.util.ArrayList;
import util.CommandLineParser;
import util.SOPLUtil;

/**
 * This class implements a recently proposed unsupervised approach for choosing
 * the optimal exponent to use for calculating Minkowski distances in the data.
 * The approach is based on selecting either the exponent leading to the lowest
 * anti-hub rate or the lowest hub rate, as they are correlated. This approach
 * was proposed in the paper titled "Choosing the Metric in High-Dimensional
 * Spaces Based on Hub Analysis" by Dominik Schnitzer and Arthur Flexer that was
 * presented at the 22nd European Symposium on Artificial Neural Networks,
 * Computational Intelligence and Machine Learning in 2014. In that paper, hubs
 * were formally defined as points that occur at least twice as often as
 * expected, which is somewhat less formal and flexible than using the more
 * standard approach of marking points whose occurrence counts exceed the
 * average by at least two standard deviations as hubs. Nevertheless, for
 * consistency with the paper, we use the Nk(x) >= 2k as a criterion for hubs in
 * this class.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MinkowskiDegreeAutoFinder {

    // Whether hub rates or anti-hub rates are to be used in selection.
    public enum DegreeSelectionCriterion {
        
        HUB, ANTIHUB
    }
    private static final int DEFAULT_NUM_THREADS = 8;
    private static final int DEFAULT_NEIGHBORHOOD_SIZE = 1;
    // Data to learn the best exponent for.
    private DataSet dset;
    // The range where to search for the best exponent.
    private float minExp = 0.25f;
    private float maxExp = 4f;
    private float stepExp = 0.25f;
    // The current best distance matrix.
    private float[][] bestMatrix;
    // The current best exponent.
    private float bestExponent;
    // Lists of calculated parameters.
    private ArrayList<Float> testedExponents;
    private ArrayList<Float> antiHubRates;
    private ArrayList<Float> hubRates;
    // The selection criterion.
    private DegreeSelectionCriterion selectionCriterion =
            DegreeSelectionCriterion.ANTIHUB;
    // Number of threads to use for distance matrix calculations.
    private int numThreads = DEFAULT_NUM_THREADS;
    private int k = DEFAULT_NEIGHBORHOOD_SIZE;
    private boolean verbose = false;

    /**
     * Initialization.
     *
     * @param dset DataSet to learn the optimal Minkowski exponent for.
     */
    public MinkowskiDegreeAutoFinder(DataSet dset) {
        if (dset == null || dset.isEmpty()) {
            throw new IllegalArgumentException("DataSet must not be empty.");
        }
        this.dset = dset;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to learn the optimal Minkowski exponent for.
     * @param minExp Float value that is the minimum exponent to try.
     * @param maxExp Float value that is the maximum exponent to try.
     * @param stepExp Float value that is the step for the exponent search.
     */
    public MinkowskiDegreeAutoFinder(DataSet dset, float minExp, float maxExp,
            float stepExp) {
        if (dset == null || dset.isEmpty()) {
            throw new IllegalArgumentException("DataSet must not be empty.");
        }
        this.dset = dset;
        if (minExp > maxExp) {
            throw new IllegalArgumentException("Maximum exponent lower than "
                    + "minimum exponent. Error.");
        }
        this.minExp = minExp;
        this.maxExp = maxExp;
        this.stepExp = stepExp;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to learn the optimal Minkowski exponent for.
     * @param selectionCriterion DegreeSelectionCriterion used for determining
     * the optimal exponent.
     */
    public MinkowskiDegreeAutoFinder(DataSet dset,
            DegreeSelectionCriterion selectionCriterion) {
        if (dset == null || dset.isEmpty()) {
            throw new IllegalArgumentException("DataSet must not be empty.");
        }
        this.dset = dset;
        this.selectionCriterion = selectionCriterion;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet to learn the optimal Minkowski exponent for.
     * @param minExp Float value that is the minimum exponent to try.
     * @param maxExp Float value that is the maximum exponent to try.
     * @param stepExp Float value that is the step for the exponent search.
     * @param selectionCriterion DegreeSelectionCriterion used for determining
     * the optimal exponent.
     */
    public MinkowskiDegreeAutoFinder(DataSet dset, float minExp, float maxExp,
            float stepExp, DegreeSelectionCriterion selectionCriterion) {
        if (dset == null || dset.isEmpty()) {
            throw new IllegalArgumentException("DataSet must not be empty.");
        }
        this.dset = dset;
        if (minExp > maxExp) {
            throw new IllegalArgumentException("Maximum exponent lower than "
                    + "minimum exponent. Error.");
        }
        this.minExp = minExp;
        this.maxExp = maxExp;
        this.stepExp = stepExp;
        this.selectionCriterion = selectionCriterion;
    }

    /**
     * @return ArrayList<Float> of all tested Minkowski exponents.
     */
    public ArrayList<Float> listAllTestedExponents() {
        return testedExponents;
    }

    /**
     * @return ArrayList<Float> of all computed anti-hub rates for the tested
     * exponents.
     */
    public ArrayList<Float> listAntiHubRates() {
        return antiHubRates;
    }

    /**
     * @return ArrayList<Float> of all computed hub occurrence rates for the
     * tested exponents.
     */
    public ArrayList<Float> listHubRates() {
        return hubRates;
    }

    /**
     * @param selectionCriterion DegreeSelectionCriterion used for determining
     * the optimal exponent.
     */
    public void setSelectionCriterion(
            DegreeSelectionCriterion selectionCriterion) {
        this.selectionCriterion = selectionCriterion;
    }

    /**
     * @param numThreads Integer that is the number of threads to use for
     * distance calculations.
     */
    public void setNumThreads(int numThreads) {
        this.numThreads = numThreads;
    }

    /**
     * @param k Integer value representing the neighborhood size to use in the
     * calculations.
     */
    public void setNeighborhoodSize(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("Neighborhood size must be "
                    + "positive.");
        }
        this.k = k;
    }

    /**
     * @return float[][] representing the upper triangular distance matrix that
     * corresponds to the best calculated Minkowski exponent value.
     */
    public float[][] getBestDistanceMatrix() {
        return bestMatrix;
    }

    /**
     * @return Float value that was determined to be the best according to the
     * selected hubness-aware criterion.
     */
    public float getBestExponent() {
        return bestExponent;
    }

    /**
     * @return CombinedMetric object corresponding to the best computed
     * Minkowski exponent.
     */
    public CombinedMetric getBestMetricObject() {
        MinkowskiMetric met = new MinkowskiMetric(bestExponent);
        CombinedMetric cmet =
                new CombinedMetric(met, met, CombinedMetric.Mixer.SUM);
        return cmet;
    }

    /**
     * @param verbose Boolean value indicating whether to print out which
     * exponent is currently being tested during iterations. This is meant to be
     * used for script executions of the evaluation and the default false value
     * should be used when the code is invoked from other classes in some
     * internal calculations.
     */
    public void setVerboseMode(boolean verbose) {
        this.verbose = verbose;
    }

    /**
     * This method calculates the hub and anti-hub rates for the specified
     * Minkowski exponent range.
     */
    public void examineParameterRange() throws Exception {
        CombinedMetric cmet;
        MinkowskiMetric met;
        NeighborSetFinder nsf;
        float[][] dMat;
        float minimalAntiHubRate = Float.MAX_VALUE;
        float minimalHubRate = Float.MAX_VALUE;
        // Initialize the result lists.
        int rangeNum = (int) Math.ceil((maxExp - minExp) / stepExp) + 1;
        testedExponents = new ArrayList<>(rangeNum);
        antiHubRates = new ArrayList<>(rangeNum);
        hubRates = new ArrayList<>(rangeNum);
        // Iterate.
        for (float expVal = minExp; DataMineConstants.isPositive(
                (maxExp - expVal)); expVal += stepExp) {
            if (verbose) {
                System.out.println("Testing for exponent value: " + expVal);
            }
            // Initialize the metric objects.
            met = new MinkowskiMetric(expVal);
            cmet = new CombinedMetric(met, met, CombinedMetric.Mixer.SUM);
            // Calculate the distance matrix.
            dMat = dset.calculateDistMatrixMultThr(cmet, numThreads);
            // Calculate the neighbor sets.
            nsf = new NeighborSetFinder(dset, dMat, cmet);
            nsf.calculateNeighborSetsMultiThr(k, numThreads);
            // Get the neighbor occurrence frequencies.
            int[] occFreqs = nsf.getNeighborFrequencies();
            // As proposed in the paper, so we follow the same convention.
            int hubThreshold = 2 * k;
            // Again, following what was proposed in the paper.
            int antiHubThreshold = 0;
            float numAntiHubs = 0;
            float sumHubOccurrences = 0;
            for (int i = 0; i < dset.size(); i++) {
                if (occFreqs[i] >= hubThreshold) {
                    sumHubOccurrences += occFreqs[i];
                }
                if (occFreqs[i] <= antiHubThreshold) {
                    numAntiHubs++;
                }
            }
            // Calculate the hub rate and anti-hub rate.
            float antiHubRate = numAntiHubs / dset.size();
            float hubRate = sumHubOccurrences / (dset.size() * k);
            // Log the results.
            testedExponents.add(expVal);
            antiHubRates.add(antiHubRate);
            hubRates.add(hubRate);
            // Update the minimal hub and anti-hub rates.
            if (antiHubRate < minimalAntiHubRate) {
                minimalAntiHubRate = antiHubRate;
                if (selectionCriterion == DegreeSelectionCriterion.ANTIHUB) {
                    bestExponent = expVal;
                    bestMatrix = dMat;
                }
            }
            if (hubRate < minimalHubRate) {
                minimalHubRate = hubRate;
                if (selectionCriterion == DegreeSelectionCriterion.HUB) {
                    bestExponent = expVal;
                    bestMatrix = dMat;
                }
            }
            System.gc();
        }
    }

    /**
     * A script that utilizes the implemented method to find the optimal
     * Minkowski exponent on a specified dataset, for which the user provides a
     * path from the command line. Users can also specify the exponent ranges,
     * as well as the neighborhood size and the selection criterion.
     *
     * @param args String[] representing the command line parameters, as
     * specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input dataset.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-k", "Neighborhood size.", CommandLineParser.INTEGER,
                true, false);
        clp.addParam("-selectionCriterion", "'hub' or 'antihub' (no quotes).",
                CommandLineParser.STRING, true, false);
        clp.addParam("-expMin", "Minimal exponent to try.",
                CommandLineParser.FLOAT, true, false);
        clp.addParam("-expMax", "Maximal exponent to try.",
                CommandLineParser.FLOAT, true, false);
        clp.addParam("-expStep", "Exponent step to use during search.",
                CommandLineParser.FLOAT, true, false);
        clp.parseLine(args);
        DataSet dset = SupervisedLoader.loadData(
                (String) (clp.getParamValues("-inFile").get(0)), false);
        float expMin = (Float) (clp.getParamValues("-expMin").get(0));
        float expMax = (Float) (clp.getParamValues("-expMax").get(0));
        float expStep = (Float) (clp.getParamValues("-expStep").get(0));
        MinkowskiDegreeAutoFinder finder =
                new MinkowskiDegreeAutoFinder(dset, expMin, expMax, expStep);
        int k = (Integer) (clp.getParamValues("-k").get(0));
        String selCriterionString = (String) (clp.getParamValues(
                "-selectionCriterion").get(0));
        finder.setNeighborhoodSize(k);
        if (selCriterionString.equalsIgnoreCase("hub")) {
            finder.setSelectionCriterion(DegreeSelectionCriterion.HUB);
        } else if (selCriterionString.equalsIgnoreCase("antihub")) {
            finder.setSelectionCriterion(DegreeSelectionCriterion.ANTIHUB);
        } else {
            throw new Exception("Bad selection criterion specification. Must be"
                    + " 'hub' or 'antihub' (no quotes). Provided string was: "
                    + selCriterionString);
        }
        finder.setNumThreads(DEFAULT_NUM_THREADS);
        finder.setVerboseMode(true);
        finder.examineParameterRange();
        System.out.println("Best exponent " + finder.getBestExponent());
        System.out.println("Computed hub rates:");
        SOPLUtil.printArrayList(finder.listHubRates());
        System.out.println("Computed antihub rates:");
        SOPLUtil.printArrayList(finder.listAntiHubRates());
        System.out.println("List of tested exponent values:");
        SOPLUtil.printArrayList(finder.listAllTestedExponents());
    }
}

/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package learning.supervised.example;

import data.representation.DataSet;
import distances.primary.CombinedMetric;
import ioformat.SupervisedLoader;
import java.io.File;
import learning.supervised.evaluation.ClassificationEstimator;
import learning.supervised.methods.knn.AKNN;
import util.CommandLineParser;

/**
 * This class gives a usage example for classification - how to load the data,
 * train a classification model and save the results to a file. As an example,
 * the adaptive k-nearest neighbor classifier (AKNN) is used.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ClassifierUsageExample {

    public static void main(String[] args) throws Exception {
        // Specify the command line parameters. While it is possible to write
        // custom command line parsing methods for each class, the utility
        // CommandLineParser class makes it easy in HubMiner.
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFileTrain", "Path to the input training data file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-inFileTest", "Path to the input test data file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Path to the output file.",
                CommandLineParser.STRING, true, false);
        // The parser parses the command line to extract the parameter values.
        clp.parseLine(args);
        // We assign the in/out train and test file path values to the
        // respective variables.
        File inFileTrain =
                new File((String) clp.getParamValues("-inFileTrain").get(0));
        File inFileTest =
                new File((String) clp.getParamValues("-inFileTest").get(0));
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        // Data load is simple when done via the SupervisedLoader class. It
        // can handle .arff, .csv and .tsv files. It can also load data from the
        // sparse modifications of the .arff format that are used in HubMiner.
        // It detects and loads the proper format automatically.
        DataSet datasetTrain = SupervisedLoader.loadData(inFileTrain, false);
        DataSet datasetTest = SupervisedLoader.loadData(inFileTest, false);
        // We use a default metric here, the Euclidean distance.
        CombinedMetric cmet = CombinedMetric.EUCLIDEAN;
        // We choose a desired neighborhood size.
        int k = 5;
        // Initialization of the classifier.
        AKNN classifier = new AKNN(datasetTrain, cmet, k);
        // Model training.
        classifier.train();
        // An aggregate ClassificationEstimator object is generated when the
        // predictions are compared on the test set.
        ClassificationEstimator estimator = classifier.test(datasetTest);
        // The estimator values are output to a file.
        estimator.printEstimatorToFile(outFile);
    }
}

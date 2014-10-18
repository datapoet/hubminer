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
package ioformat.util;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.sparse.BOWDataSet;
import ioformat.IOARFF;
import ioformat.IOCSV;
import ioformat.SupervisedLoader;
import java.io.File;
import util.CommandLineParser;

/**
 * This class implements a utility script for generating several separate
 * labeled data feature files from a single sparse data representation and
 * multiple possible labeling options that are given in a separate .csv file.
 * This is sometimes useful for clean separate testing, though it takes more
 * disk space.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class LabelMultiAttacher {

    /**
     * This script generates the labeled data files from an unlabeled data
     * representation and multiple label arrays given in the label .csv file.
     *
     * @param args Command line arguments.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-representationInput", "Path to the input data file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-inLabels", "Path to the labels csv file.",
                CommandLineParser.STRING, true, false);
        clp.parseLine(args);
        File inFile = new File((String) clp.getParamValues(
                "-representationInput").get(0));
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        File inLabelFile = new File((String) clp.getParamValues("-inLabels").
                get(0));
        DataSet dset = SupervisedLoader.loadData(inFile, true);
        boolean isSparse = dset instanceof BOWDataSet;
        // Now read the labels.
        IOCSV reader = new IOCSV(false, " +");
        DataSet labelDSet = reader.readData(inLabelFile);
        int numLabelArrays = labelDSet.getNumFloatAttr();
        DataInstance repInstance, labelInstance;
        String dsetName = inFile.getName();
        if (dsetName.lastIndexOf('.') > 0) {
            dsetName = dsetName.substring(0, dsetName.lastIndexOf('.'));
        }
        for (int labelOption = 0; labelOption < numLabelArrays; labelOption++) {
            for (int index = 0; index < dset.size(); index++) {
                repInstance = dset.data.get(index);
                labelInstance = labelDSet.data.get(index);
                repInstance.setCategory((int) (labelInstance.fAttr[
                        labelOption]));
            }
            // Generate the file for the current labeling.
            IOARFF persister = new IOARFF();
            if (isSparse) {
                persister.saveSparseLabeled((BOWDataSet) dset,
                        (new File(outDir, dsetName + "l" + labelOption
                        + ".arff")).getPath());
            } else {
                persister.saveLabeledWithIdentifiers(dset,
                        (new File(outDir, dsetName + "l" + labelOption
                        + ".arff")).getPath(), null);
            }
        }
    }
}

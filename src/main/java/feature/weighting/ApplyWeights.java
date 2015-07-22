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
package feature.weighting;

import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.IOARFF;
import java.io.File;
import util.CommandLineParser;

/**
 * Applies feature weights to a dataset.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ApplyWeights {

    /**
     * Applies feature weights to the specified dataset.
     *
     * @param dset DataSet to apply the weights to.
     * @param weightsDSet Weights DataSet to apply.
     */
    public static void apply(DataSet dset, DataSet weightsDSet) {
        if (dset == null || weightsDSet == null || dset.isEmpty()
                || weightsDSet.isEmpty()) {
            return;
        }
        if (dset.getNumFloatAttr() < weightsDSet.getNumFloatAttr()
                && dset.getNumIntAttr() >= weightsDSet.getNumFloatAttr()) {
            for (DataInstance instance : dset.data) {
                for (int d = 0; d < weightsDSet.getNumFloatAttr(); d++) {
                    instance.iAttr[d] *= weightsDSet.getInstance(0).fAttr[d];
                }
            }
        } else {
            for (DataInstance instance : dset.data) {
                for (int d = 0; d < weightsDSet.getNumFloatAttr(); d++) {
                    instance.fAttr[d] *= weightsDSet.getInstance(0).fAttr[d];
                }
            }
        }
    }

    /**
     * Applies weights to the specified dataset.
     *
     * @param args Command line parameters: input data, input weights and the
     * output path.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inDataSet", "Path to the dataset",
                CommandLineParser.STRING, true, false);
        clp.addParam("-inWeights", "Path to the weights DataSet",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output path", CommandLineParser.STRING,
                true, false);
        clp.parseLine(args);
        File inDataSet = new File(
                (String) (clp.getParamValues("-inDataSet").get(0)));
        File outFile = new File(
                (String) (clp.getParamValues("-outFile").get(0)));
        File inWeights = new File(
                (String) (clp.getParamValues("-inWeights").get(0)));
        IOARFF reader = new IOARFF();
        DataSet inSet = reader.load(inDataSet.getPath());
        DataSet weights = reader.load(inWeights.getPath());
        apply(inSet, weights);
        IOARFF saver = new IOARFF();
        saver.saveLabeled(inSet, outFile.getPath());
    }
}

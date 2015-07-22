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

import data.representation.sparse.BOWDataSet;
import ioformat.IOARFF;
import java.io.File;
import util.CommandLineParser;

/**
 * This utility script loads in the sparse matrix formatted data and transforms
 * it into the sparse ARFF format as used in this library.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class IOSparseMatrix {

    /**
     * Reads in one sparse data format and writes the other one.
     *
     * @param inFile Input file.
     * @param outFile Output file.
     * @param offset Integer that is the instance index offset, the difference
     * in indexing protocols.
     * @param featOffset Integer that is the feature index offset, the
     * difference in indexing protocols.
     * @throws Exception
     */
    public static void transformSparseFormats(File inFile, File outFile,
            int offset, int featOffset) throws Exception {
        if (inFile == null || !inFile.exists() || !inFile.isFile()) {
            throw new Exception("Bad input path specified.");
        }
        BOWDataSet dset = new BOWDataSet();
        dset.loadFromSparseMatrixFile(inFile.getPath(), offset, featOffset);
        IOARFF persister = new IOARFF();
        persister.saveSparseUnlabeled(dset, outFile.getPath());
    }

    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input sparse file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Path to the output sparse file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-docOffset", "Integer that is the instance index offset, "
                + "the difference in indexing protocols.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-featureOffset", "Integer that is the feature index "
                + "offset, the difference in indexing protocols.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File inFile = new File((String) clp.getParamValues(
                "-inFile").get(0));
        File outFile = new File((String) clp.getParamValues("-outFile").get(0));
        int docOffset = (Integer) clp.getParamValues("-docOffset").get(0);
        int featureOffset = (Integer) clp.getParamValues("-featureOffset").
                get(0);
        transformSparseFormats(inFile, outFile, docOffset, featureOffset);
    }
}

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
package util.text;

import data.representation.sparse.BOWInstance;
import data.representation.sparse.BOWDataSet;
import data.representation.DataSet;
import data.representation.DataInstance;
import ioformat.IOARFF;
import java.io.File;
import java.util.ArrayList;

/**
 * A utility class for loading a BOW dataset from an arff file and prints it
 * into a sparse matrix format.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class BOWLoaderFromARFF {

    /**
     * Loads the BOW DataSet from the input.
     *
     * @param inFile Input file containing the BOW.
     * @return BOWDataSet object.
     * @throws Exception
     */
    public static BOWDataSet load(File inFile) throws Exception {
        IOARFF persister = new IOARFF();
        DataSet dset = persister.load(inFile.getPath());
        BOWDataSet bowdset = new BOWDataSet();
        bowdset.data = new ArrayList<>(dset.size());
        BOWInstance tempBOW;
        DataInstance temp;
        String[] tokens;
        for (int i = 0; i < dset.size(); i++) {
            temp = dset.data.get(i);
            tempBOW = new BOWInstance(bowdset);
            tempBOW.documentName = temp.sAttr[0];
            temp.sAttr[1] = temp.sAttr[1].trim();
            tokens = temp.sAttr[1].split(" ");
            if (tokens != null) {
                for (int j = 0; j < tokens.length; j++) {
                    tempBOW.addWord(tokens[j]);
                }
            }
            bowdset.addDataInstance(tempBOW);
        }
        return bowdset;
    }

    /**
     * Read a BOW arff and print to a sparse matrix file.
     *
     * @param args Command line arguments. The first parameter is the input file
     * and the second is the output file.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("arg0: inFile (.arff of a BOW DataSet)");
            System.out.println("arg1: outFile (sparse matrix format)");
        }
        BOWDataSet bdset = load(new File(args[0]));
        bdset.removeLessFrequentWords(6);
        bdset.printToSparseMatrixFile(args[1]);
    }
}

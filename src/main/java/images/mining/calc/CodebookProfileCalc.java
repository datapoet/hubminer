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
package images.mining.calc;

import data.representation.DataInstance;
import data.representation.DataSet;
import images.mining.codebook.SIFTCodeBook;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

/**
 * A utility class for calculating a codebook profile for all codebook features.
 * The idea is that different visual words occur with different probabilities in
 * images of different classes in the data and this script can be used to
 * quickly extract and analyze the visual word occurrence profiles.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CodebookProfileCalc {

    private File inFileHistograms, inFileCodebook, outFile;
    private double[][] codebookProfiles;
    SIFTCodeBook codebook;
    int numClasses;

    /**
     * Initialization.
     *
     * @param inFileHistograms File that contains the quantized image
     * representations.
     * @param inFileCodebook File that contains the codebook definition.
     * @param outFile File for printing out the results of the analysis.
     */
    public CodebookProfileCalc(File inFileHistograms, File inFileCodebook,
            File outFile) {
        this.inFileCodebook = inFileCodebook;
        this.inFileHistograms = inFileHistograms;
        this.outFile = outFile;
    }

    /**
     * Calculates codebook profiles from the data that was provided.
     *
     * @throws Exception
     */
    public void generateCodebookProfiles() throws Exception {
        IOARFF arff = new IOARFF();
        DataSet siftBOW = arff.load(inFileHistograms.getPath());
        numClasses = siftBOW.countCategories();
        codebook = new SIFTCodeBook();
        codebook.loadCodeBookFromFile(inFileCodebook);
        codebookProfiles = new double[codebook.getSize()][numClasses];
        for (int i = 0; i < siftBOW.size(); i++) {
            int currClass = siftBOW.getLabelOf(i);
            DataInstance instance = siftBOW.getInstance(i);
            for (int codeIndex = 0; codeIndex < codebook.getSize();
                    codeIndex++) {
                if (instance != null && instance.iAttr != null) {
                    codebookProfiles[codeIndex][currClass] +=
                            instance.iAttr[codeIndex];
                }
            }
        }
    }

    /**
     * Saves the results of the analysis.
     *
     * @throws Exception
     */
    public void saveCodebookProfiles() throws Exception {
        FileUtil.createFile(outFile);
        PrintWriter pw = new PrintWriter(new FileWriter(outFile));
        try {
            pw.println(codebook.getSize());
            for (int codeIndex = 0; codeIndex < codebook.getSize();
                    codeIndex++) {
                pw.print(codebookProfiles[codeIndex][0]);
                for (int c = 1; c < numClasses; c++) {
                    pw.print("," + codebookProfiles[codeIndex][c]);
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        } finally {
            pw.close();
        }
    }

    /**
     * Prints out command line info.
     */
    public static void info() {
        System.out.println("arg0: histogram dataset, arg1: codebookFile, "
                + "arg2: outputFile");
    }

    /**
     * A script that calculates the codebook occurrence profiles.
     *
     * @param args Command line arguments, as specified in the info method,
     * input files for the quantized image representation, codebook file... and
     * output for the analysis.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            info();
            return;
        }
        CodebookProfileCalc cpc = new CodebookProfileCalc(new File(args[0]),
                new File(args[1]), new File(args[2]));
        cpc.generateCodebookProfiles();
        cpc.saveCodebookProfiles();
    }
}

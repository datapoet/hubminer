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
package learning.supervised.evaluation.cv;

import com.google.gson.Gson;
import ioformat.FileUtil;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import util.ReaderToStringUtil;

/**
 * This class implements the methods for persisting and loading the
 * cross-validation folds.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CVFoldsIO {

    private static class FoldsObject {

        ArrayList<Integer>[][] allFolds;
        int numTimes;
        int numFolds;

        /**
         * Default constructor.
         */
        public FoldsObject() {
        }

        /**
         * Initialization.
         *
         * @param allFolds ArrayList<Integer>[][] representing the folds.
         * @param numTimes Integer that is the number of repetitions of the fold
         * procedure in this cross-validation.
         * @param numFolds Integer that is the number of splits (folds) in the
         * cross-validation procedure.
         */
        public FoldsObject(ArrayList<Integer>[][] allFolds, int numTimes,
                int numFolds) {
            this.allFolds = allFolds;
            this.numTimes = numTimes;
            this.numFolds = numFolds;
        }
    }

    /**
     * Saves the cross-validation folds to a file.
     *
     * @param allFolds ArrayList<Integer>[][] representing the folds.
     * @param numTimes Integer that is the number of repetitions of the fold
     * procedure in this cross-validation.
     * @param numFolds Integer that is the number of splits (folds) in the
     * cross-validation procedure.
     * @param outFile File to write the cross-validation folds to.
     */
    public static void saveAllFolds(ArrayList<Integer>[][] allFolds,
            int numTimes, int numFolds, File outFile) throws Exception {
        if (allFolds == null) {
            return;
        }
        if (numTimes <= 0 || numFolds <= 0) {
            return;
        }
        FileUtil.createFile(outFile);
        Gson gson = new Gson();
        FoldsObject fObj = new FoldsObject(allFolds, numTimes, numFolds);
        String jsonString = gson.toJson(fObj, FoldsObject.class);
        try (PrintWriter pw = new PrintWriter(new FileWriter(outFile))) {
            pw.write(jsonString);
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * Loads the cross-validation folds from a file. This can be used to run
     * multiple rounds of tests on same data folds.
     *
     * @param inFile File to load the cross-validation folds from.
     * @return
     */
    public static ArrayList<Integer>[][] loadAllFolds(File inFile)
            throws Exception {
        if (!inFile.exists() || !inFile.isFile()) {
            throw new Exception("Invalid fold load file path.");
        }
        String jsonString = new String(Files.readAllBytes(
                Paths.get(inFile.getPath())));
        Gson gson = new Gson();
        FoldsObject fObj = gson.fromJson(jsonString, FoldsObject.class);
        return fObj.allFolds;
    }
    
    /**
     * Saves the cross-validation folds to a file.
     *
     * @param allFolds ArrayList<Integer>[][] representing the folds.
     * @param numTimes Integer that is the number of repetitions of the fold
     * procedure in this cross-validation.
     * @param numFolds Integer that is the number of splits (folds) in the
     * cross-validation procedure.
     * @param outFile File to write the cross-validation folds to.
     */
    public static void saveAllFolds(ArrayList<Integer>[][] allFolds,
            int numTimes, int numFolds, Writer writer) throws Exception {
        if (allFolds == null) {
            return;
        }
        if (numTimes <= 0 || numFolds <= 0) {
            return;
        }
        if (writer == null) {
            throw new Exception("Empty writer provided.");
        }
        Gson gson = new Gson();
        FoldsObject fObj = new FoldsObject(allFolds, numTimes, numFolds);
        String jsonString = gson.toJson(fObj, FoldsObject.class);
        writer.write(jsonString);
    }

    /**
     * Loads the cross-validation folds from a file. This can be used to run
     * multiple rounds of tests on same data folds.
     *
     * @param inFile File to load the cross-validation folds from.
     * @return
     */
    public static ArrayList<Integer>[][] loadAllFolds(Reader reader)
            throws Exception {
        if (reader == null) {
            throw new Exception("Empty reader provided.");
        }
        String jsonString = ReaderToStringUtil.readAsSingleString(reader);
        Gson gson = new Gson();
        FoldsObject fObj = gson.fromJson(jsonString, FoldsObject.class);
        return fObj.allFolds;
    }
}

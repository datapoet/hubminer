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
package ioformat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.io.Writer;

/**
 * This class handles some basic IO operations for reading and writing the
 * distance matrix from and to files in the format that is used throughout this
 * library.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DistanceMatrixIO {

    /**
     * Print the distance matrix of the currently considered dataset to a file
     * for later loading.
     *
     * @param distMat float[][] that is the upper triangular distance matrix.
     * @param dMatFile File that is to contain the distance matrix data.
     * @throws Exception
     */
    public static void printDMatToFile(float[][] distMat, File dMatFile)
            throws Exception {
        FileUtil.createFile(dMatFile);
        try (PrintWriter pw = new PrintWriter(new FileWriter(dMatFile));) {
            pw.println(distMat.length); // Data size.
            for (int i = 0; i < distMat.length - 1; i++) {
                // The last line is empy anyway, as distMat is upper Triangular
                // matrix with a zero diagonal.
                pw.print(distMat[i][0]);
                for (int j = 1; j < distMat[i].length; j++) {
                    pw.print("," + distMat[i][j]);
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * Load the upper triangular distance matrix from a file. This is done when
     * the distance matrix had already been calculated in the past, in order to
     * avoid needless repetitive calculations.
     *
     * @param dMatFile File containing the distance matrix data.
     * @return float[][] that is the loaded distance matrix.
     * @throws Exception
     */
    public static float[][] loadDMatFromFile(File dMatFile) throws Exception {
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(dMatFile)));
        float[][] dMatLoaded = null;
        String line;
        String[] lineParse;
        try {
            int size = Integer.parseInt(br.readLine());
            dMatLoaded = new float[size][];
            for (int i = 0; i < size - 1; i++) {
                dMatLoaded[i] = new float[size - i - 1];
                line = br.readLine();
                lineParse = line.split(",");
                for (int j = 0; j < lineParse.length; j++) {
                    dMatLoaded[i][j] = Float.parseFloat(lineParse[j]);
                }
            }
            dMatLoaded[size - 1] = new float[0];
        } catch (IOException | NumberFormatException e) {
            throw e;
        } finally {
            br.close();
        }
        return dMatLoaded;
    }
    
    /**
     * Print the distance matrix of the currently considered dataset to a file
     * for later loading.
     *
     * @param distMat float[][] that is the upper triangular distance matrix.
     * @param writer Writer to print the distance matrix data to.
     * @throws Exception
     */
    public static void printDMatToStream(float[][] distMat, Writer writer)
            throws Exception {
        try (PrintWriter pw = new PrintWriter(writer)) {
            pw.println(distMat.length); // Data size.
            for (int i = 0; i < distMat.length - 1; i++) {
                // The last line is empy anyway, as distMat is upper Triangular
                // matrix with a zero diagonal.
                pw.print(distMat[i][0]);
                for (int j = 1; j < distMat[i].length; j++) {
                    pw.print("," + distMat[i][j]);
                }
                pw.println();
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * Load the upper triangular distance matrix from a file. This is done when
     * the distance matrix had already been calculated in the past, in order to
     * avoid needless repetitive calculations.
     *
     * @param reader Reader to load the distance matrix from.
     * @return float[][] that is the loaded distance matrix.
     * @throws Exception
     */
    public static float[][] loadDMatFromStream(Reader reader) throws Exception {
        float[][] dMatLoaded = null;
        String line;
        String[] lineParse;
        try (BufferedReader br = new BufferedReader(reader)) {
            int size = Integer.parseInt(br.readLine());
            dMatLoaded = new float[size][];
            for (int i = 0; i < size - 1; i++) {
                dMatLoaded[i] = new float[size - i - 1];
                line = br.readLine();
                lineParse = line.split(",");
                for (int j = 0; j < lineParse.length; j++) {
                    dMatLoaded[i][j] = Float.parseFloat(lineParse[j]);
                }
            }
            dMatLoaded[size - 1] = new float[0];
        } catch (IOException | NumberFormatException e) {
            throw e;
        }
        return dMatLoaded;
    }
}

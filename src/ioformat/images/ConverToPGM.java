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
package ioformat.images;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

/**
 * This class is a proxy for PGM conversion done by ImageMagick.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ConverToPGM {

    /**
     * Performs batch JPEG to PGM conversion.
     *
     * @param inputDirectory Input JPG directory.
     * @param outputDirectory Output PGM directory.
     * @throws Exception
     */
    public static void convertFolder(File inputDirectory,
            File outputDirectory) throws Exception {
        File[] children = inputDirectory.listFiles();
        File subDir;
        if (children != null) {
            for (int i = 0; i < children.length; i++) {
                if (children[i].isFile()) {
                    convertFile(children[i],
                            new File(outputDirectory.getPath()
                            + File.separator + children[i].getName().substring(
                            0, children[i].getName().length() - 3) + "pgm"));
                } else {
                    subDir = new File(outputDirectory, children[i].getName());
                    if (!(subDir.exists())) {
                        subDir.mkdir();
                    }
                    convertFolder(children[i], subDir);
                }
            }
        }
    }

    /**
     * Converts a JPG file to a PGM file.
     *
     * @param inputFile Input JPG file path.
     * @param outputFile Output PGM file path.
     * @throws Exception
     */
    public static void convertFile(File inputFile, File outputFile)
            throws Exception {
        String parameterString = "";
        callImageMagick(inputFile.getPath(), outputFile.getPath(),
                parameterString);
    }

    /**
     * Converts a JPG file to a PGM file.
     *
     * @param inputPath Input JPG file path.
     * @param outputPath Output PGM file path.
     * @param paramsString String with additional parameters for the ImageMagick
     * external call.
     * @throws Exception
     */
    private static void callImageMagick(String inputPath, String outputPath,
            String paramsString) throws Exception {
        String cline = "cmd /c convert" + paramsString + " " + inputPath
                + " " + outputPath;
        Runtime rt = Runtime.getRuntime();
        Process proc = rt.exec(cline, null, null);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                     proc.getInputStream()))) {
            proc.getErrorStream().close();
            int exitVal = proc.waitFor();
            proc.destroy();
            String s = br.readLine();
        }
    }
}
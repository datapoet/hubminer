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
import java.util.Stack;

/**
 * Utility methods for creating and copying files, so that it is not a problem
 * when the parent of the target does not exist.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class FileUtil {

    /**
     * This method copies the provided file.
     *
     * @param original File that is the original file.
     * @param copy File that is the copy of the original file.
     * @throws Exception
     */
    public static void copyFile(File original, File copy) throws Exception {
        if (!original.exists()) {
            return;
        }
        createFile(copy);
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(original)));
        PrintWriter pw = new PrintWriter(new FileWriter(copy));
        try {
            String line = br.readLine();
            while (line != null) {
                pw.println(line);
                line = br.readLine();
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            br.close();
            pw.close();
        }
    }

    /**
     * Creates the target file, regardless of whether its parent directories
     * exist. It creates them if they don't.
     *
     * @param targetPath String that is the path of the target file to create.
     * @throws IOException
     */
    public static void createFileFromPath(String targetPath)
            throws IOException {
        createFile(new File(targetPath));
    }

    /**
     * Creates the target file, regardless of whether its parent directories
     * exist. It creates them if they don't.
     *
     * @param targetFile File that is the target to create.
     * @throws IOException
     */
    public static void createFile(File targetFile) throws IOException {
        if (!targetFile.exists()) {
            File targetParent;
            Stack parents = new Stack();
            targetParent = targetFile.getParentFile();
            while (targetParent != null) {
                parents.push(targetParent);
                targetParent = targetParent.getParentFile();
            }
            while (!parents.empty()) {
                targetParent = (File) parents.pop();
                if (!targetParent.exists()) {
                    targetParent.mkdir();
                }
            }
            targetFile.createNewFile();
        } else {
            targetFile.delete();
            targetFile.createNewFile();
        }
    }

    /**
     * Creates the target directory, regardless of whether its parent
     * directories exist. It creates them if they don't.
     *
     * @param targetDir File that is the target directory to create.
     * @throws IOException
     */
    public static void createDirectory(File targetDir) throws IOException {
        if (!targetDir.exists()) {
            Stack parents = new Stack();
            File targetParent = targetDir.getParentFile();
            while (targetParent != null) {
                parents.push(targetParent);
                targetParent = targetParent.getParentFile();
            }
            while (!parents.empty()) {
                targetParent = (File) parents.pop();
                if (!targetParent.exists()) {
                    targetParent.mkdir();
                }
            }
            targetDir.mkdir();
        } else {
            targetDir.delete();
            targetDir.mkdir();
        }
    }

    /**
     * Fetch file names recursively and print them to a stream.
     *
     * @param dirPath Directory to fetch the names from.
     * @param pw PrintWriter that is the stream to write the names to.
     * @throws Exception
     */
    public static void fetchFileNamesRecursively(String dirPath, PrintWriter pw)
            throws Exception {
        File dir = new File(dirPath);
        if (dir.exists() && dir.isDirectory()) {
            File[] children = dir.listFiles();
            if (children != null) {
                for (int i = 0; i < children.length; i++) {
                    if (children[i].isFile()) {
                        pw.println(children[i].getName());
                    } else {
                        fetchFileNamesRecursively(children[i].getPath(), pw);
                    }
                }
            }
        }
    }
}
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

import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.IOARFF;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.HashMap;

/**
 * A utility class for extracting text from a specified field in a csv file.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CSVTextGrabber {

    // The indexes of fields that contain text and names.
    int nameFieldIndex = -1;
    int textFieldIndex = -1;
    String[] nameList = null;
    HashMap<String, Integer> nameMap = null;
    String separator = " ";

    /**
     * @param nameFieldIndex Integer that is the name field index.
     * @param textFieldIndex Integer that is the text field index.
     */
    public CSVTextGrabber(int nameFieldIndex, int textFieldIndex) {
        this.nameFieldIndex = nameFieldIndex;
        this.textFieldIndex = textFieldIndex;
    }

    /**
     * Initialization.
     *
     * @param nameList String[] that contains the list of names to consider.
     * @param nameFieldIndex Integer that is the name field index.
     * @param textFieldIndex Integer that is the text field index.
     */
    public CSVTextGrabber(
            String[] nameList,
            int nameFieldIndex,
            int textFieldIndex) {
        this.nameList = nameList;
        if (nameList != null) {
            nameMap = new HashMap(nameList.length * 2);
            for (int i = 0; i < nameList.length; i++) {
                nameMap.put(nameList[i], i);
            }
        }
        this.nameFieldIndex = nameFieldIndex;
        this.textFieldIndex = textFieldIndex;
    }

    /**
     * Checks if a name is in the name list.
     *
     * @param name String that is the name to check for.
     * @return True if in the list, false otherwise.
     */
    public boolean inNameList(String name) {
        return nameMap.containsKey(name);
    }

    /**
     * Checks if a name is in the name list, but without the quotes.
     *
     * @param name String that is the name to check for.
     * @return True if in the list, false otherwise.
     */
    public boolean inNameListUnquoted(String name) {
        String unquotedName;
        if (name.length() > 2) {
            unquotedName = name.substring(1, name.length() - 1);
        } else {
            unquotedName = name;
        }
        return nameMap.containsKey(unquotedName);
    }

    /**
     * Sets the acceptable name list.
     *
     * @param nameList String[] that is the acceptable name list.
     */
    public void setNameList(String[] nameList) {
        this.nameList = nameList;
        if (nameList != null) {
            nameMap = new HashMap(nameList.length * 2);
            for (int i = 0; i < nameList.length; i++) {
                nameMap.put(nameList[i], i);
            }
        }
    }

    /**
     * Sets the default separator to use.
     *
     * @param separator String that is the separator.
     */
    public void setSeparator(String separator) {
        this.separator = separator;
    }

    /**
     * Grab the text from the specified fields in the specified csv file.
     *
     * @param path File path to the input file.
     * @return DataSet object with grabbed text.
     * @throws Exception
     */
    public DataSet grabText(String path) throws Exception {
        return grabText(new File(path));
    }

    /**
     * Grab the text from the specified fields in the specified csv file.
     *
     * @param inputFile Input file.
     * @return DataSet object with grabbed text.
     * @throws Exception
     */
    public DataSet grabText(File inputFile) throws Exception {
        if (!inputFile.exists()) {
            return null;
        }
        if (nameList == null || nameList.length == 0) {
            return null;
        }
        String s;
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(inputFile)));
        DataSet dc = new DataSet();
        dc.sAttrNames = new String[2];
        dc.sAttrNames[0] = "name";
        dc.sAttrNames[1] = "text";
        String[] lineParse;
        String name;
        String text;
        DataInstance tempInstance;
        try {
            s = br.readLine();
            while (s != null) {
                try {
                    lineParse = s.split(separator);
                    name = lineParse[nameFieldIndex];
                    if (inNameListUnquoted(name) || inNameList(name)) {
                        text = lineParse[textFieldIndex];
                        tempInstance = new DataInstance(dc);
                        tempInstance.sAttr[0] = name;
                        if (!text.startsWith("Translation in progress")) {
                            tempInstance.sAttr[1] = text;
                        } else {
                            tempInstance.sAttr[1] = "";
                        }
                        tempInstance.embedInDataset(dc);
                        dc.addDataInstance(tempInstance);
                    }
                } catch (Exception e2) {
                    System.err.println(e2.getMessage());
                }
                s = br.readLine();
            }
        } catch (Exception e) {
            throw e;
        } finally {
            br.close();
        }
        return dc;
    }

    /**
     * Grab the text and write it to a file.
     *
     * @param inPath Path to the input file.
     * @param outPath Path to the output file.
     * @throws Exception
     */
    public void grabAndWriteToFile(String inPath, String outPath)
            throws Exception {
        grabAndWriteToFile(new File(inPath), new File(outPath));
    }

    /**
     * Grab the text and write it to a file.
     *
     * @param inFile Input file.
     * @param outFile Output file.
     * @throws Exception
     */
    public void grabAndWriteToFile(File inFile, File outFile) throws Exception {
        DataSet dc = grabText(inFile);
        IOARFF persister = new IOARFF();
        persister.save(dc, outFile.getPath(), null);
    }

    /**
     * Command line parameter specification.
     */
    public static void info() {
        System.out.println("Gonna grab the textual annotations out of CSV");
        System.out.println("Index in the range 0 : (numFields-1) needs to be "
                + "specified for both the name field and text field");
        System.out.println("Output will be written in arff format");
        System.out.println("List of selected names needs to be given in a "
                + "separate file to specify what to choose");
        System.out.println("In that file - each name is in the new line, and "
                + "the first line in the file contains the number of names");
        System.out.println("arg0: name field index");
        System.out.println("arg1: text field index");
        System.out.println("arg2: input path");
        System.out.println("arg3: name list path");
        System.out.println("arg4: output path");
    }

    /**
     * Extracts the text and name fields from specified fields in the specified
     * csv file and prints them out in the arff file format to the output path
     * that was given.
     *
     * @param args Command line parameters.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            info();
        } else {
            CSVTextGrabber grabber = new CSVTextGrabber(
                    Integer.parseInt(args[0]), Integer.parseInt(args[1]));
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(
                    new FileInputStream(new File(args[3]))));
            String s;
            String[] names = new String[0];
            int numNames = 0;
            try {
                s = br.readLine();
                s = s.trim();
                try {
                    numNames = Integer.parseInt(s);
                } catch (Exception e1) {
                    System.err.println(e1.getMessage());
                }
                names = new String[numNames];
                for (int i = 0; i < numNames; i++) {
                    s = br.readLine();
                    s = s.trim();
                    names[i] = s;
                }
            } catch (Exception e) {
                throw e;
            } finally {
                br.close();
            }
            grabber.setNameList(names);
            System.out.println(names.length + " instances");
            grabber.separator = ",";
            grabber.grabAndWriteToFile(new File(args[2]), new File(args[4]));
            System.out.println("Task completed...");
        }
    }
}

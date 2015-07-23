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

import ioformat.IOARFF;
import data.representation.DataSet;
import java.io.File;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.util.ArrayList;
import data.representation.DataInstance;

/**
 * This utility class can either take a file of names and a file of associated
 * text and generate a DataSet of both, or de-couple an existing DataSet that
 * has names as its first attribute and text as its second into two files, one
 * corresponding to names, the other to text.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NamesTextCouplingUtil {

    /**
     * Takes the original file that has names as its first attribute, like text
     * titles - and text as its second - and breaks it up into two files.
     *
     * @param originalFile ARFF file that has names as its first attribute, like
     * text titles - and text as its second.
     * @param nameFile ARFF file of names/titles.
     * @param textFile ARFF file of textual content.
     * @throws Exception
     */
    public static void splitFile(
            File originalFile,
            File nameFile,
            File textFile) throws Exception {
        IOARFF persister = new IOARFF();
        DataSet dset = persister.load(originalFile.getPath());
        DataSet dsetNames, dcText;
        dsetNames = new DataSet();
        dcText = new DataSet();
        dsetNames.sAttrNames = new String[1];
        dsetNames.sAttrNames[0] = "name";
        dcText.sAttrNames = new String[1];
        dcText.sAttrNames[0] = "text";
        dsetNames.data = new ArrayList<>(dset.size());
        DataInstance instance;
        for (int i = 0; i < dset.size(); i++) {
            instance = new DataInstance(dsetNames);
            instance.embedInDataset(dsetNames);
            instance.sAttr[0] = dset.data.get(i).sAttr[0];
            dsetNames.addDataInstance(instance);
            instance = new DataInstance(dcText);
            instance.embedInDataset(dcText);
            instance.sAttr[0] = dset.data.get(i).sAttr[1];
            dcText.addDataInstance(instance);
        }
        persister.save(dsetNames, nameFile.getPath(), null);
        persister.save(dcText, textFile.getPath(), null);
    }

    /**
     * Combines a file of names/titles and a file of associated text into a
     * single DataSet and persists it to a new ARFF file.
     *
     * @param nameFile File of names/titles - each in a new line.
     * @param textFile File of textual content - each in a new line.
     * @param newFile ARFF file that has names as its first attribute, like text
     * titles - and text as its second.
     * @throws Exception
     */
    public static void join(File nameFile, File textFile, File newFile)
            throws Exception {
        DataSet dset = new DataSet();
        dset.data = new ArrayList<>(2000);
        dset.sAttrNames = new String[2];
        dset.sAttrNames[0] = "name";
        dset.sAttrNames[1] = "text";
        BufferedReader brName = new BufferedReader(
                new InputStreamReader(new FileInputStream(nameFile)));
        BufferedReader brText = new BufferedReader(
                new InputStreamReader(new FileInputStream(textFile)));
        DataInstance instance;
        String sName, sText;
        try {
            sName = brName.readLine();
            sText = brText.readLine();
            while (sName != null && sText != null) {
                instance = new DataInstance(dset);
                instance.sAttr[0] = sName.trim();
                instance.sAttr[1] = sText.trim();
                dset.addDataInstance(instance);
                sName = brName.readLine();
                sText = brText.readLine();
            }
        } catch (Exception e) {
            throw e;
        } finally {
            brName.close();
            brText.close();
        }
        IOARFF persister = new IOARFF();
        persister.save(dset, newFile.getPath(), null);
    }

    /**
     * Information about input parameters.
     */
    public void info() {
        System.out.println("args0: SPLIT or JOIN");
        System.out.println("In case of JOIN:");
        System.out.println(" - args1: Names file, each in a new line.");
        System.out.println(" - args2: Text file, each in a new line.");
        System.out.println(" - args3: Combined file, ARFF.");
        System.out.println("In case of SPLIT:");
        System.out.println(" - args1: Combined file, ARFF.");
        System.out.println(" - args2: Names file, ARFF.");
        System.out.println(" - args3: Text file, ARFF.");
    }

    /**
     * @param args Command line arguments, as described in the info() method.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            return;
        }
        if (args[0].equalsIgnoreCase("SPLIT")) {
            splitFile(new File(args[1]), new File(args[2]), new File(args[3]));
        } else if (args[0].equalsIgnoreCase("JOIN")) {
            join(new File(args[1]), new File(args[2]), new File(args[3]));
        } else {
            System.out.println("Wrong util command.");
        }
    }
}

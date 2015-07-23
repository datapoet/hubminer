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
package graph.io;

import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import graph.basic.DMGraph;
import graph.basic.DMGraphEdge;
import graph.basic.VertexInstance;
import ioformat.FileUtil;
import java.awt.Color;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.charset.Charset;
import java.text.DateFormat;
import java.util.Date;

/**
 * This class implements the IO methods for writing and reading Pajek graph
 * files.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PajekFormatIO {

    /**
     * Writes the graph to a Pajek-format file.
     *
     * @param g DMGraph that is the graph to write.
     * @param outPath String that is the path to persist the graph to.
     * @param featureType Integer representing the feature type for display.
     * @param featureIndex Integer representing the feature index to display.
     * @throws Exception
     */
    public static void writeGraphToPajekFile(DMGraph g, String outPath,
            int featureType, int featureIndex) throws Exception {
        if (g == null || g.isEmpty()) {
            return;
        }
        File outFile = new File(outPath);
        FileUtil.createFile(outFile);
        try (final PrintWriter pw = new PrintWriter(new OutputStreamWriter(
                new FileOutputStream(outFile), Charset.forName("UTF8")))) {
            Date currentTime = new Date();
            pw.println("*Network " + g.networkName);
            pw.println("%Transformed to Pajek format on "
                    + DateFormat.getDateTimeInstance().format(currentTime));
            pw.println("*Vertices " + g.getNumberOfVertices());
            featureType = featureType % 3;
            for (int vertexIndex = 0; vertexIndex < g.size(); vertexIndex++) {
                VertexInstance vi = (VertexInstance) (g.vertices.data.get(
                        vertexIndex));
                switch (featureType) {
                    case DataMineConstants.NOMINAL:
                        pw.println((vertexIndex + 1) + " \"" + vi.sAttr[
                                featureIndex] + "\" ellipse ic "
                                + colorToPajekColor(vi.colour)
                                + " bc Black x_fact " + Math.max(1.0, vi.scale)
                                + " y_fact " + Math.max(1.0, vi.scale));
                        break;
                    case DataMineConstants.INTEGER:
                        pw.println((vertexIndex + 1) + " \""
                                + (new Integer(vi.iAttr[featureIndex])).
                                toString() + "\" ellipse ic "
                                + colorToPajekColor(vi.colour)
                                + " bc Black x_fact " + Math.max(1.0, vi.scale)
                                + " y_fact " + Math.max(1.0, vi.scale));
                        break;
                    case DataMineConstants.FLOAT:
                        pw.println((vertexIndex + 1) + " \"" + (new Float(
                                vi.fAttr[featureIndex])).toString()
                                + "\" ellipse ic "
                                + colorToPajekColor(vi.colour)
                                + " bc Black x_fact " + Math.max(1.0, vi.scale)
                                + " y_fact " + Math.max(1.0, vi.scale));
                        break;
                    default:
                        pw.println((vertexIndex + 1) + " \"" + vi.sAttr[
                                featureIndex] + "\" ellipse ic "
                                + colorToPajekColor(vi.colour)
                                + " bc Black x_fact " + Math.max(1.0, vi.scale)
                                + " y_fact " + Math.max(1.0, vi.scale));
                        break;
                }
            }
            pw.println("*Arcs");
            pw.println("*Edges");
            DMGraphEdge edge;
            for (int i = 0; i < g.edges.length; i++) {
                edge = g.edges[i];
                while (edge != null) {
                    if (edge.first < edge.second) {
                        pw.println((edge.first + 1) + " " + (edge.second + 1)
                                + " " + edge.weight);
                    }
                    edge = edge.next;
                }
            }
        } catch (Exception e) {
            throw e;
        }
    }

    /**
     * A primitive method to distinguish red/blue/green for Pajek. In Pajek,
     * there is no mechanism to assign an RGB color and it would take too long
     * to make a complete mapping between the corresponding color names.
     *
     * @param col Color object.
     * @return String that is the name of the color to be inserted in the Pajek
     * file.
     * @throws Exception
     */
    static String colorToPajekColor(Color col) throws Exception {
        int red = col.getRed();
        int blue = col.getBlue();
        int green = col.getGreen();
        if ((red >= blue) && (red >= green)) {
            return "Red";
        } else if ((blue >= red) && (blue >= green)) {
            return "Blue";
        } else {
            return "Green";
        }
    }

    /**
     * This method loads a DMGraph object from a Pajek file.
     *
     * @param inPath String that is the input path.
     * @return DMGraph that is loaded from the file.
     * @throws Exception
     */
    public static DMGraph loadGraphFromPajekFile(String inPath)
            throws Exception {
        File inFile = new File(inPath);
        if (!(inFile.exists() && inFile.isFile())) {
            throw new Exception("Bad input path. File does not exist: "
                    + inPath);
        }
        DMGraph g = new DMGraph();
        BufferedReader br = new BufferedReader(new InputStreamReader(
                new FileInputStream(inFile), Charset.forName("UTF8")));
        // Different file reading modes.
        boolean vertexMode = false;
        boolean edgeMode = false;
        boolean verticesDone = false;
        boolean edgesDone = false;
        DMGraphEdge edge;
        VertexInstance vertex;
        String[] tokens;
        try {
            String line = br.readLine();
            while (line != null) {
                line = line.trim();
                if (line.startsWith("*Network")) {
                    line = line.substring(8, line.length());
                    line = line.trim();
                    g.networkName = line;
                } else if (line.startsWith("*Vertices")) {
                    if (!verticesDone) {
                        line = line.substring(9, line.length());
                        line = line.trim();
                        Integer tmpInteger = new Integer(line);
                        int numVertices = tmpInteger.intValue();
                        g.edges = new DMGraphEdge[numVertices];
                        g.vertices = new DataSet(numVertices);
                        g.vertices.sAttrNames = new String[1];
                        g.vertices.sAttrNames[0] = "Vertex name";
                        vertexMode = true;
                    }
                    if (edgeMode) {
                        edgesDone = true;
                    }
                    edgeMode = false;
                } else if (line.startsWith("*Edges")) {
                    if (vertexMode) {
                        verticesDone = true;
                    }
                    vertexMode = false;
                    if (!edgesDone) {
                        edgeMode = true;
                    }
                } else if (line.startsWith("*Arcs")) {
                    if (vertexMode) {
                        verticesDone = true;
                    }
                    if (edgeMode) {
                        edgesDone = true;
                    }
                    vertexMode = false;
                    edgeMode = false;
                } else if (line.startsWith("*Partition")) {
                    if (vertexMode) {
                        verticesDone = true;
                    }
                    if (edgeMode) {
                        edgesDone = true;
                    }
                    vertexMode = false;
                    edgeMode = false;
                } else {
                    if (vertexMode) {
                        vertex = new VertexInstance(g.vertices);
                        tokens = line.split("[ ]+");
                        boolean vertexNameFinished = false;
                        vertex.sAttr[0] = tokens[1];
                        if ((tokens[1].charAt(0) == '\'') && (
                                tokens[1].charAt(tokens[1].length() - 1)
                                == '\'')) {
                            vertexNameFinished = true;
                        }
                        for (int tokenIndex = 0; tokenIndex < tokens.length;
                                tokenIndex++) {
                            if ((!vertexNameFinished) && (tokenIndex >= 2)) {
                                vertex.sAttr[0] = vertex.sAttr[0].concat(" ");
                                vertex.sAttr[0] = vertex.sAttr[0].concat(
                                        tokens[tokenIndex]);
                                if (tokens[tokenIndex].charAt(
                                        tokens[tokenIndex].length() - 1)
                                        == '\'') {
                                    vertexNameFinished = true;
                                }
                            }
                            switch (tokens[tokenIndex]) {
                                case "ic": {
                                    Field field = Class.forName(
                                            "java.awt.Color").getField(
                                            (tokens[tokenIndex + 1]).
                                            toUpperCase());
                                    vertex.colour = (Color) field.get(null);
                                    break;
                                }
                                case "x_fact": {
                                    vertex.scale = (new Double(tokens[
                                                tokenIndex + 1])).doubleValue();
                                    break;
                                }
                                case "y_fact": {
                                    vertex.scale = (new Double(tokens[
                                                tokenIndex + 1])).doubleValue();
                                    break;
                                }
                            }
                        }
                        vertex.sAttr[0] = vertex.sAttr[0].substring(1,
                                vertex.sAttr[0].length() - 1);
                        g.vertices.addDataInstance(vertex);
                    }
                    if (edgeMode) {
                        edge = new DMGraphEdge();
                        tokens = line.split("[ ]+");
                        try {
                            edge.first = (new Integer(tokens[0])).intValue()
                                    - 1;
                            edge.second = (new Integer(tokens[1])).intValue()
                                    - 1;
                            edge.weight = (new Double(tokens[2])).doubleValue();
                            DMGraph.insertEdge(g.edges, edge);
                            edge = new DMGraphEdge();
                            edge.first = (new Integer(tokens[1])).intValue()
                                    - 1;
                            edge.second = (new Integer(tokens[0])).intValue()
                                    - 1;
                            edge.weight = (new Double(tokens[2])).doubleValue();
                            DMGraph.insertEdge(g.edges, edge);
                        } catch (Exception e) {
                            System.err.println(e.getMessage());
                        }
                    }
                }
                line = br.readLine();
            }
        } catch (IOException | ClassNotFoundException | NoSuchFieldException |
                SecurityException | IllegalArgumentException |
                IllegalAccessException e) {
            throw e;
        } finally {
            br.close();
        }
        return g;
    }
}

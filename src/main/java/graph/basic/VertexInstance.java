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
package graph.basic;

import data.representation.DataInstance;
import data.representation.DataSet;
import java.awt.Color;
import java.util.Arrays;
import org.jgraph.graph.DefaultGraphCell;

/**
 * This class represents graph vertices.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class VertexInstance extends DataInstance {

    // Display properties should ideally be separated from the data model, so
    // this should probably be re-organized. The data that corresponds to the
    // node is contained in the underlying DataInstance, in the feature arrays.
    public static final Color DEFAULT_COLOR = Color.BLUE;
    public Color colour;
    // Scale.
    public double scale;
    // Coordinates.
    public double x, y;
    // Optional, if drawing by JGraph.
    public DefaultGraphCell jgVertex = null;

    /**
     * The default constructor.
     */
    public VertexInstance() {
        super();
        this.colour = DEFAULT_COLOR;
    }

    /**
     * Initialization.
     *
     * @param instance DataInstance that holds the data of this vertex node.
     */
    public VertexInstance(DataInstance instance) {
        this.embedInDataset(instance.getEmbeddingDataset());
        this.fAttr = instance.fAttr;
        this.iAttr = instance.iAttr;
        this.sAttr = instance.sAttr;
        this.scale = 1.;
        this.x = 0;
        this.y = 0;
        this.jgVertex = null;
        this.colour = DEFAULT_COLOR;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet data context to initialize the node data with.
     */
    public VertexInstance(DataSet dset) {
        super(dset);
        this.colour = DEFAULT_COLOR;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet data context to initialize the node data with.
     * @param scale Double value that is the vertex scale.
     */
    public VertexInstance(DataSet dset, double scale) {
        super(dset);
        this.scale = scale;
        this.colour = DEFAULT_COLOR;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet data context to initialize the node data with.
     * @param scale Double value that is the vertex scale.
     * @param x Double that is the X coordinate.
     * @param y
     */
    public VertexInstance(DataSet dset, double scale, double x, double y) {
        super(dset);
        this.scale = scale;
        this.x = x;
        this.y = y;
        this.colour = DEFAULT_COLOR;
    }

    /**
     * Initialization.
     *
     * @param dset DataSet data context to initialize the node data with.
     * @param scale Double value that is the vertex scale.
     * @param x Double that is the X coordinate.
     * @param y Double that is the Y coordinate.
     * @param colour Color that is the vertex color.
     */
    public VertexInstance(DataSet dset, double scale, double x, double y,
            Color colour) {
        super(dset);
        this.scale = scale;
        this.x = x;
        this.y = y;
        this.colour = colour;
    }

    @Override
    public VertexInstance copy() {
        VertexInstance newInstance;
        try {
            newInstance = (VertexInstance) (super.copy());
        } catch (Exception eFirst) {
            newInstance = new VertexInstance(getEmbeddingDataset());
            // Copy the data arrays.
            if (getEmbeddingDataset().hasIntAttr()) {
                newInstance.iAttr = Arrays.copyOf(iAttr, iAttr.length);
            }
            if (getEmbeddingDataset().hasFloatAttr()) {
                newInstance.fAttr = Arrays.copyOf(fAttr, fAttr.length);
            }
            if (getEmbeddingDataset().hasNominalAttr()) {
                newInstance.sAttr = Arrays.copyOf(sAttr, sAttr.length);
            }
            if (getIdentifier() != null) {
                try {
                    newInstance.setIdentifier(getIdentifier().copy());
                } catch (Exception eSecond) {
                    newInstance.setIdentifier(null);
                }
            } else {
                newInstance.setIdentifier(null);
            }
        }
        // Copy the rendering information.
        newInstance.colour = colour;
        newInstance.scale = scale;
        newInstance.x = x;
        newInstance.y = y;
        return newInstance;
    }

    @Override
    public VertexInstance copyContent() {
        VertexInstance newInstance;
        try {
            newInstance = (VertexInstance) (super.copy());
        } catch (Exception e) {
            newInstance = new VertexInstance(getEmbeddingDataset());
            // Copy the data arrays.
            if (getEmbeddingDataset().hasIntAttr()) {
                newInstance.iAttr = Arrays.copyOf(iAttr, iAttr.length);
            }
            if (getEmbeddingDataset().hasFloatAttr()) {
                newInstance.fAttr = Arrays.copyOf(fAttr, fAttr.length);
            }
            if (getEmbeddingDataset().hasNominalAttr()) {
                newInstance.sAttr = Arrays.copyOf(sAttr, sAttr.length);
            }
        }
        // Copy the rendering information.
        newInstance.colour = colour;
        newInstance.scale = scale;
        newInstance.x = x;
        newInstance.y = y;
        return newInstance;
    }
}
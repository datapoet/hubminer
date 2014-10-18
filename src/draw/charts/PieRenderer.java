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
package draw.charts;

import java.awt.Color;
import java.util.List;
import org.jfree.chart.plot.PiePlot;
import org.jfree.chart.plot.PiePlot3D;
import org.jfree.data.general.DefaultPieDataset;

/**
 * Pie renderer class that knows how to deal with pie charts in jfreechart.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PieRenderer {

    private Color[] colors;

    /**
     * @param colors Color array to use for coloring the chart.
     */
    public PieRenderer(Color[] colors) {
        this.colors = colors;
    }

    /**
     * Iterates through the Pie dataset keys and assigns proper colors to their
     * segments.
     *
     * @param plot PiePlot object.
     * @param dataset DefaultPieDataset that the charts is drawn from.
     */
    public void setColor(PiePlot plot, DefaultPieDataset dataset) {
        List<Comparable> keys = dataset.getKeys();
        int colorIndex;
        for (int i = 0; i < keys.size(); i++) {
            colorIndex = i % this.colors.length;
            plot.setSectionPaint(keys.get(i), this.colors[colorIndex]);
        }
    }

    /**
     * Iterates through the Pie dataset keys and assigns proper colors to their
     * segments.
     *
     * @param plot PiePlot3D object.
     * @param dataset DefaultPieDataset that the charts is drawn from.
     */
    public void setColor(PiePlot3D plot, DefaultPieDataset dataset) {
        List<Comparable> keys = dataset.getKeys();
        int colorIndex;
        for (int i = 0; i < keys.size(); i++) {
            colorIndex = i % this.colors.length;
            plot.setSectionPaint(keys.get(i), this.colors[colorIndex]);
        }
    }
}

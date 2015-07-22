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
package gui.images;

import draw.charts.PieRenderer;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import javax.swing.JFrame;
import javax.swing.JPanel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PiePlot;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.util.Rotation;

/**
 * This panel is used for displaying a visual word occurrence profile in a
 * chart, along with its index, while estimating the utility of different visual
 * words.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CodebookVectorProfilePanel extends javax.swing.JPanel {

    private int codebookIndex = -1;
    private double[] occurrenceProfile = null;

    /**
     * Creates new form CodebookVectorProfilePanel
     */
    public CodebookVectorProfilePanel() {
        initComponents();
    }

    /**
     * @return Integer that is the index of the represented codebook.
     */
    public int getCodebookIndex() {
        return codebookIndex;
    }

    /**
     * Generate a BufferedImage that would correspond to a JPanel. Images are
     * faster to show than interactive components if many components need to be
     * presented.
     *
     * @param panel JPanel object.
     * @return BufferedImage of how the content in the provided panel would be
     * rendered.
     */
    public BufferedImage createImage(JPanel panel) {
        int width = panel.getSize().width;
        int height = panel.getSize().height;
        BufferedImage bi = new BufferedImage(width, height,
                BufferedImage.TYPE_INT_RGB);
        Graphics2D g = bi.createGraphics();
        panel.paint(g);
        return bi;
    }

    /**
     * Sets the data to be shown.
     *
     * @param occurrenceProfile Double array that is the neighbor occurrence
     * profile of this visual word.
     * @param codebookIndex Integer that is the index of this visual word.
     * @param classColors Color[] of class colors.
     * @param classNames String[] of class names.
     */
    public void setResults(double[] occurrenceProfile, int codebookIndex,
            Color[] classColors, String[] classNames) {
        int numClasses = Math.min(classNames.length, occurrenceProfile.length);
        this.codebookIndex = codebookIndex;
        this.occurrenceProfile = occurrenceProfile;
        DefaultPieDataset pieData = new DefaultPieDataset();
        for (int cIndex = 0; cIndex < numClasses; cIndex++) {
            pieData.setValue(classNames[cIndex], occurrenceProfile[cIndex]);
        }
        JFreeChart chart = ChartFactory.createPieChart3D("codebook vect "
                + codebookIndex, pieData, true, true, false);
        PiePlot plot = (PiePlot) chart.getPlot();
        plot.setDirection(Rotation.CLOCKWISE);
        plot.setForegroundAlpha(0.5f);
        PieRenderer prend = new PieRenderer(classColors);
        prend.setColor(plot, pieData);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(140, 140));
        chartPanel.setVisible(true);
        chartPanel.revalidate();
        chartPanel.repaint();
        JPanel jp = new JPanel();
        jp.setPreferredSize(new Dimension(140, 140));
        jp.setMinimumSize(new Dimension(140, 140));
        jp.setMaximumSize(new Dimension(140, 140));
        jp.setSize(new Dimension(140, 140));
        jp.setLayout(new FlowLayout());
        jp.add(chartPanel);
        jp.setVisible(true);
        jp.validate();
        jp.repaint();

        JFrame frame = new JFrame();
        frame.setBackground(Color.WHITE);
        frame.setUndecorated(true);
        frame.getContentPane().add(jp);
        frame.pack();
        BufferedImage bi = new BufferedImage(jp.getWidth(), jp.getHeight(),
                BufferedImage.TYPE_INT_ARGB);
        Graphics2D graphics = bi.createGraphics();
        jp.print(graphics);
        graphics.dispose();
        frame.dispose();
        imPanel.removeAll();
        imPanel.setImage(bi);
        imPanel.setVisible(true);
        imPanel.revalidate();
        imPanel.repaint();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        imPanel = new gui.images.ImagePanel();

        imPanel.setMaximumSize(new java.awt.Dimension(140, 140));
        imPanel.setMinimumSize(new java.awt.Dimension(140, 140));
        imPanel.setPreferredSize(new java.awt.Dimension(140, 140));

        javax.swing.GroupLayout imPanelLayout = new javax.swing.GroupLayout(imPanel);
        imPanel.setLayout(imPanelLayout);
        imPanelLayout.setHorizontalGroup(
            imPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 140, Short.MAX_VALUE)
        );
        imPanelLayout.setVerticalGroup(
            imPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 140, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(imPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(imPanel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
    }// </editor-fold>//GEN-END:initComponents
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private gui.images.ImagePanel imPanel;
    // End of variables declaration//GEN-END:variables
}

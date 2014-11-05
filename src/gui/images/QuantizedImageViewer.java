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

import data.representation.images.sift.LFeatRepresentation;
import data.representation.images.sift.LFeatVector;
import images.mining.codebook.SIFTCodeBook;
import images.mining.display.SIFTDraw;
import java.awt.Color;
import java.awt.Component;
import java.awt.FlowLayout;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import javax.imageio.ImageIO;
import javax.swing.JFileChooser;
import javax.swing.JFrame;

/**
 * This frame is used for visual word utility assessment in Image Hub Explorer.
 * It shows a list of visual words with their class-conditional occurrence
 * profiles and it shows the utility landscape on top of the current image. It
 * is possible to examine the total image region utility, as well as to
 * visualize each visual word separately.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class QuantizedImageViewer extends javax.swing.JFrame {

    // The relevant data sources. Not all of them are directly used from within
    // this frame in the current implementation, but the functionality of this
    // frame is open to many extensions and it will be extended soon. Therefore,
    // a bit more information was included to begin with.
    // Feature representation for the image in question.
    private LFeatRepresentation imageSIFT;
    private float[] codebookGoodness;
    // An object that represents the visual word definitions.
    private SIFTCodeBook codebook;
    private double[][] codebookProfiles;
    // Class colors for display.
    private Color[] classColors;
    // Class names for display.
    private String[] classNames;
    // Currently examined image.
    private BufferedImage originalImage;
    // An image of the overall utility of different regions in the original
    // image.
    private BufferedImage goodnessSIFTImage;
    // Each image in the array corresponds to a visualization of a single
    // visual word.
    private BufferedImage[] codebookVisualizationImages;
    // This array holds the index of the closest codebook feature for each
    // feature in the currently examined image.
    private int[] codebookAssignments;
    // One partial representation for each codebook feature.
    private LFeatRepresentation[] partialReps;
    // Black and white image.
    private BufferedImage bwImage;
    private File currentDirectory = new File(".");
    // Visual word index of the currently examined codebook feature.
    private int selectedCodebookFeatureIndex = -1;

    /**
     * Creates new form QuantizedImageViewer
     */
    public QuantizedImageViewer() {
        initComponents();
    }

    /**
     * Initialization.
     *
     * @param originalImage BufferedImage that is the current image.
     * @param imageSIFT Image feature representation.
     * @param codebookGoodness Float array of codebook goodness scores.
     * @param codebook SIFTCodeBook object that holds the visual word
     * definitions.
     * @param codebookProfiles double[][] that represents the class-conditional
     * occurrence profiles for all the codebooks.
     * @param cProfPanels CodebookVectorProfilePanel[] of the codebook profile
     * panels.
     * @param classColors Color[] of class colors.
     * @param classNames String[] of class names.
     */
    public QuantizedImageViewer(
            BufferedImage originalImage,
            LFeatRepresentation imageSIFT,
            float[] codebookGoodness,
            SIFTCodeBook codebook,
            double[][] codebookProfiles,
            CodebookVectorProfilePanel[] cProfPanels,
            Color[] classColors,
            String[] classNames) {
        initComponents();
        codebookProfilesPanel.setLayout(new FlowLayout());
        this.imageSIFT = imageSIFT;
        this.codebookGoodness = codebookGoodness;
        this.codebook = codebook;
        this.codebookProfiles = codebookProfiles;
        this.classColors = classColors;
        this.classNames = classNames;
        this.originalImage = originalImage;
        // Get a proper black-white image to draw on.
        bwImage = new BufferedImage(originalImage.getWidth(),
                originalImage.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = bwImage.createGraphics();
        g2d.drawImage(originalImage, 0, 0, originalImage.getWidth(),
                originalImage.getHeight(), null);
        BufferedImage bwImageTmp = new BufferedImage(originalImage.getWidth(),
                originalImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
        g2d = bwImageTmp.createGraphics();
        g2d.drawImage(bwImage, 0, 0, originalImage.getWidth(),
                originalImage.getHeight(), null);
        bwImage = bwImageTmp;
        partialReps = new LFeatRepresentation[codebook.getSize()];
        codebookVisualizationImages = new BufferedImage[codebook.getSize()];
        // Insert all the individual codebook profile visualization panels.
        for (int cInd = 0; cInd < codebook.getSize(); cInd++) {
            codebookProfilesPanel.add(cProfPanels[cInd]);
            cProfPanels[cInd].addMouseListener(new CodebookSelectionListener());
            partialReps[cInd] = new LFeatRepresentation();
        }
        codebookProfilesPanel.revalidate();
        codebookProfilesPanel.repaint();
        codebookScrollPane.revalidate();
        codebookScrollPane.repaint();
        originalImagePanel.setImage(originalImage);
        // Calculate the closest codebook feature for each feature in the
        // original image.
        float[] featureGoodness = new float[imageSIFT.size()];
        codebookAssignments = new int[imageSIFT.size()];
        for (int i = 0; i < imageSIFT.size(); i++) {
            LFeatVector sv = (LFeatVector) (imageSIFT.getInstance(i));
            try {
                codebookAssignments[i] = codebook.getIndexOfClosestCodebook(sv);
                partialReps[codebookAssignments[i]].addDataInstance(sv);
            } catch (Exception e) {
                System.out.println(e);
            }
            featureGoodness[i] = codebookGoodness[codebookAssignments[i]];
        }
        // Determine the best visual word and visualize it first by default.
        int maxRepIndex = 0;
        int maxSize = 0;
        for (int cInd = 0; cInd < codebook.getSize(); cInd++) {
            if (partialReps[cInd].size() > maxSize) {
                maxRepIndex = cInd;
                maxSize = partialReps[cInd].size();
            }
        }
        visualizeVisualWordUtility(maxRepIndex);
        // Visualize the overall utility of different image regions.
        try {
            goodnessSIFTImage = SIFTDraw.drawSIFTGoodnessOnImage(imageSIFT,
                    featureGoodness, bwImage);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
        allQuantizedPanel.setImage(goodnessSIFTImage);
        setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
    }

    /**
     * Visualizes the utility of a particular visual word.
     *
     * @param codebookfeatureIndex Integer that is the index of the particular
     * codebook feature.
     */
    private void visualizeVisualWordUtility(int codebookfeatureIndex) {
        occCountValueLabel.setText(""
                + partialReps[codebookfeatureIndex].size());
        selectedCodebookFeatureIndex = codebookfeatureIndex;
        cvectLabel.setText("Observing codebook vector: "
                + codebookfeatureIndex);
        if (codebookVisualizationImages[codebookfeatureIndex] != null) {
            // If the visualization has already been calculated, just move to
            // the appropriate object in-memory.
            selectedCodebookPanel.setImage(
                    codebookVisualizationImages[codebookfeatureIndex]);
        } else {
            // Calculate a new visualization.
            if (partialReps[codebookfeatureIndex].isEmpty()) {
                // If there are no matches for this codebook feature, just show
                // the grayscale image with no features on top.
                selectedCodebookPanel.setImage(bwImage);
                return;
            }
            // Set the local image feature goodness for each feature in this
            // partial view that contains only the matches to the current
            // visual word to be the goodness of that visual word.
            float[] featureGoodness = new float[partialReps[
                    codebookfeatureIndex].size()];
            Arrays.fill(featureGoodness, 0, featureGoodness.length,
                    codebookGoodness[codebookfeatureIndex]);
            try {
                codebookVisualizationImages[codebookfeatureIndex] =
                        SIFTDraw.drawSIFTGoodnessOnImage(
                        partialReps[codebookfeatureIndex],
                        featureGoodness, bwImage);
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            selectedCodebookPanel.setImage(
                    codebookVisualizationImages[codebookfeatureIndex]);
        }
    }

    /**
     * Listener for visual word selections.
     */
    class CodebookSelectionListener implements MouseListener {

        @Override
        public void mousePressed(MouseEvent e) {
        }

        @Override
        public void mouseReleased(MouseEvent e) {
        }

        @Override
        public void mouseEntered(MouseEvent e) {
        }

        @Override
        public void mouseExited(MouseEvent e) {
        }

        @Override
        public void mouseClicked(MouseEvent e) {
            Component comp = e.getComponent();
            System.out.println("selection made");
            if (comp instanceof CodebookVectorProfilePanel) {
                int index = ((CodebookVectorProfilePanel) comp).
                        getCodebookIndex();
                System.out.println("selected index " + index);
                visualizeVisualWordUtility(index);
            } else if (comp.getParent() != null && comp.getParent() instanceof
                    CodebookVectorProfilePanel) {
                int index = ((CodebookVectorProfilePanel) comp.getParent()).
                        getCodebookIndex();
                System.out.println("selected index " + index);
                visualizeVisualWordUtility(index);
            } else if (comp.getParent() != null && comp.getParent().getParent()
                    != null && comp.getParent().getParent() instanceof
                    CodebookVectorProfilePanel) {
                int index = ((CodebookVectorProfilePanel) comp.getParent().
                        getParent()).getCodebookIndex();
                System.out.println("selected index " + index);
                visualizeVisualWordUtility(index);
            }
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jScrollPane2 = new javax.swing.JScrollPane();
        jTable1 = new javax.swing.JTable();
        originalImagePanel = new gui.images.ImagePanel();
        allQuantizedPanel = new gui.images.ImagePanel();
        selectedCodebookPanel = new gui.images.ImagePanel();
        codebookScrollPane = new javax.swing.JScrollPane();
        codebookProfilesPanel = new javax.swing.JPanel();
        cvectLabel = new javax.swing.JLabel();
        occCountLabel = new javax.swing.JLabel();
        occCountValueLabel = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();
        jMenu1 = new javax.swing.JMenu();
        jMenu2 = new javax.swing.JMenu();
        saveOverallItem = new javax.swing.JMenuItem();
        saveSelectedItem = new javax.swing.JMenuItem();

        jTable1.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {null, null, null, null},
                {null, null, null, null},
                {null, null, null, null},
                {null, null, null, null}
            },
            new String [] {
                "Title 1", "Title 2", "Title 3", "Title 4"
            }
        ));
        jScrollPane2.setViewportView(jTable1);

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Quantized Image Viewer");
        setName("qiv"); // NOI18N

        originalImagePanel.setMaximumSize(new java.awt.Dimension(450, 300));
        originalImagePanel.setMinimumSize(new java.awt.Dimension(450, 300));
        originalImagePanel.setPreferredSize(new java.awt.Dimension(450, 300));

        javax.swing.GroupLayout originalImagePanelLayout = new javax.swing.GroupLayout(originalImagePanel);
        originalImagePanel.setLayout(originalImagePanelLayout);
        originalImagePanelLayout.setHorizontalGroup(
            originalImagePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 450, Short.MAX_VALUE)
        );
        originalImagePanelLayout.setVerticalGroup(
            originalImagePanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 305, Short.MAX_VALUE)
        );

        allQuantizedPanel.setMaximumSize(new java.awt.Dimension(450, 300));
        allQuantizedPanel.setMinimumSize(new java.awt.Dimension(450, 300));
        allQuantizedPanel.setName(""); // NOI18N
        allQuantizedPanel.setPreferredSize(new java.awt.Dimension(450, 300));

        javax.swing.GroupLayout allQuantizedPanelLayout = new javax.swing.GroupLayout(allQuantizedPanel);
        allQuantizedPanel.setLayout(allQuantizedPanelLayout);
        allQuantizedPanelLayout.setHorizontalGroup(
            allQuantizedPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 450, Short.MAX_VALUE)
        );
        allQuantizedPanelLayout.setVerticalGroup(
            allQuantizedPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 305, Short.MAX_VALUE)
        );

        selectedCodebookPanel.setMaximumSize(new java.awt.Dimension(450, 300));
        selectedCodebookPanel.setMinimumSize(new java.awt.Dimension(450, 300));
        selectedCodebookPanel.setPreferredSize(new java.awt.Dimension(450, 300));

        javax.swing.GroupLayout selectedCodebookPanelLayout = new javax.swing.GroupLayout(selectedCodebookPanel);
        selectedCodebookPanel.setLayout(selectedCodebookPanelLayout);
        selectedCodebookPanelLayout.setHorizontalGroup(
            selectedCodebookPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 450, Short.MAX_VALUE)
        );
        selectedCodebookPanelLayout.setVerticalGroup(
            selectedCodebookPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 300, Short.MAX_VALUE)
        );

        codebookScrollPane.setHorizontalScrollBarPolicy(javax.swing.ScrollPaneConstants.HORIZONTAL_SCROLLBAR_ALWAYS);
        codebookScrollPane.setVerticalScrollBarPolicy(javax.swing.ScrollPaneConstants.VERTICAL_SCROLLBAR_ALWAYS);
        codebookScrollPane.setMaximumSize(new java.awt.Dimension(900, 100));
        codebookScrollPane.setMinimumSize(new java.awt.Dimension(900, 100));
        codebookScrollPane.setPreferredSize(new java.awt.Dimension(900, 100));

        codebookProfilesPanel.setMaximumSize(new java.awt.Dimension(50000, 400));
        codebookProfilesPanel.setMinimumSize(new java.awt.Dimension(50000, 100));
        codebookProfilesPanel.setPreferredSize(new java.awt.Dimension(50000, 100));

        javax.swing.GroupLayout codebookProfilesPanelLayout = new javax.swing.GroupLayout(codebookProfilesPanel);
        codebookProfilesPanel.setLayout(codebookProfilesPanelLayout);
        codebookProfilesPanelLayout.setHorizontalGroup(
            codebookProfilesPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 50000, 50000)
        );
        codebookProfilesPanelLayout.setVerticalGroup(
            codebookProfilesPanelLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 120, Short.MAX_VALUE)
        );

        codebookScrollPane.setViewportView(codebookProfilesPanel);

        cvectLabel.setText("Observing codebook vector:");

        occCountLabel.setText("Occurrence count:");

        occCountValueLabel.setText("...");

        jMenu1.setText("Photo");
        jMenuBar1.add(jMenu1);

        jMenu2.setText("Edit");

        saveOverallItem.setText("Save Overall SIFT Distribution");
        saveOverallItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveOverallItemActionPerformed(evt);
            }
        });
        jMenu2.add(saveOverallItem);

        saveSelectedItem.setText("Save Selected Codebook Distribution");
        saveSelectedItem.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                saveSelectedItemActionPerformed(evt);
            }
        });
        jMenu2.add(saveSelectedItem);

        jMenuBar1.add(jMenu2);

        setJMenuBar(jMenuBar1);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(23, 23, 23)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(cvectLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 203, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(occCountLabel)
                                .addGap(18, 18, 18)
                                .addComponent(occCountValueLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                        .addGap(18, 18, 18)
                        .addComponent(selectedCodebookPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addComponent(codebookScrollPane, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(originalImagePanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                                .addComponent(allQuantizedPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(21, 21, 21))))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(originalImagePanel, javax.swing.GroupLayout.PREFERRED_SIZE, 305, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(allQuantizedPanel, javax.swing.GroupLayout.PREFERRED_SIZE, 305, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(codebookScrollPane, javax.swing.GroupLayout.DEFAULT_SIZE, 143, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(selectedCodebookPanel, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addGroup(layout.createSequentialGroup()
                        .addComponent(cvectLabel, javax.swing.GroupLayout.PREFERRED_SIZE, 67, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addComponent(occCountLabel, javax.swing.GroupLayout.DEFAULT_SIZE, 44, Short.MAX_VALUE)
                            .addComponent(occCountValueLabel, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    /**
     * Saves the overall visual word utility image to the disk.
     *
     * @param evt ActionEvent object.
     */
    private void saveOverallItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveOverallItemActionPerformed
        try {
            File outFile;
            JFileChooser jfc = new JFileChooser(currentDirectory);
            jfc.setDialogTitle("Select file to save the component image: ");
            jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
            int rVal = jfc.showOpenDialog(QuantizedImageViewer.this);
            if (rVal == JFileChooser.APPROVE_OPTION) {
                currentDirectory = jfc.getSelectedFile().getParentFile();
                outFile = jfc.getSelectedFile();
                try {
                    ImageIO.write(goodnessSIFTImage, "jpg", outFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("problem writing file: " + e.getMessage());
        }
    }//GEN-LAST:event_saveOverallItemActionPerformed

    /**
     * Saves a visual word utility image for a particular visual word to the
     * disk.
     *
     * @param evt ActionEvent object.
     */
    private void saveSelectedItemActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_saveSelectedItemActionPerformed
        try {
            if (selectedCodebookFeatureIndex < 0
                    || codebookVisualizationImages[
                        selectedCodebookFeatureIndex] == null) {
                System.out.println("no image selected");
                return;
            }
            File outFile;
            JFileChooser jfc = new JFileChooser(currentDirectory);
            jfc.setDialogTitle("Select file to save the component image: ");
            jfc.setFileSelectionMode(JFileChooser.FILES_ONLY);
            int rVal = jfc.showOpenDialog(QuantizedImageViewer.this);
            if (rVal == JFileChooser.APPROVE_OPTION) {
                currentDirectory = jfc.getSelectedFile().getParentFile();
                outFile = jfc.getSelectedFile();
                try {
                    ImageIO.write(codebookVisualizationImages[
                                selectedCodebookFeatureIndex], "jpg", outFile);
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("problem writing file: " + e.getMessage());
        }
    }//GEN-LAST:event_saveSelectedItemActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(QuantizedImageViewer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(QuantizedImageViewer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(QuantizedImageViewer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(QuantizedImageViewer.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            @Override
            public void run() {
                new QuantizedImageViewer().setVisible(true);
            }
        });
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private gui.images.ImagePanel allQuantizedPanel;
    private javax.swing.JPanel codebookProfilesPanel;
    private javax.swing.JScrollPane codebookScrollPane;
    private javax.swing.JLabel cvectLabel;
    private javax.swing.JMenu jMenu1;
    private javax.swing.JMenu jMenu2;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JScrollPane jScrollPane2;
    private javax.swing.JTable jTable1;
    private javax.swing.JLabel occCountLabel;
    private javax.swing.JLabel occCountValueLabel;
    private gui.images.ImagePanel originalImagePanel;
    private javax.swing.JMenuItem saveOverallItem;
    private javax.swing.JMenuItem saveSelectedItem;
    private gui.images.ImagePanel selectedCodebookPanel;
    // End of variables declaration//GEN-END:variables
}

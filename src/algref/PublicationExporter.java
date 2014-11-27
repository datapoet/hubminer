/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 *
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package algref;

import java.util.Random;

/**
 * This class implements the logic for exporting the publication info to a
 * standard format for inclusion in reports and papers.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PublicationExporter {

    private static final int MAX_LABEL_LENGTH = 25;
    private static final int RAND_LABEL_LENGTH = 4;

    /**
     * This method generates the BibTex string for the provided Publication
     * object.
     *
     * @param pub Publication to generate the BibTex string for.
     * @return String that is the BibTex representation of the provided
     * Publication object.
     */
    public static String getBibTexString(Publication pub) {
        if (pub == null) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        String label = getCitableLabel(pub);
        if (pub instanceof JournalPublication) {
            builder.append("@article{");
            builder.append(label);
            builder.append(",");
            builder.append("\n");
            builder.append("journal = {");
            builder.append(((JournalPublication) pub).getJournalName());
            builder.append("},\n");
            if (((JournalPublication) pub).getVolume() > 0) {
                builder.append("volume = {");
                builder.append(((JournalPublication) pub).getVolume());
                builder.append("},\n");
            }
            if (((JournalPublication) pub).getIssue() > 0) {
                builder.append("number = {");
                builder.append(((JournalPublication) pub).getIssue());
                builder.append("},\n");
            }
        } else if (pub instanceof ConferencePublication) {
            builder.append("@inproceedings{");
            builder.append(label);
            builder.append(",");
            builder.append("\n");
            builder.append("booktitle = {");
            String proceedingsString = "Proceedings of the "
                    + ((ConferencePublication) pub).getConferenceName();
            builder.append(proceedingsString);
            builder.append("},\n");
        } else if (pub instanceof BookChapterPublication) {
            builder.append("@incollection{");
            builder.append(label);
            builder.append(",");
            builder.append("\n");
            builder.append("booktitle = {");
            builder.append(((BookChapterPublication) pub).getBookName());
            builder.append("},\n");
        } else if (pub instanceof BookPublication) {
            builder.append("@book{");
            builder.append(label);
            builder.append(",");
            builder.append("\n");
        } else {
            // The miscellanous publication type.
            builder.append("@misc{");
            builder.append(label);
            builder.append(",");
            builder.append("\n");
        }
        // Now the common part that holds for all publication types.
        if (pub.getTitle() != null) {
            // The title.
            builder.append("title = {");
            builder.append(pub.getTitle());
            builder.append("},\n");
        }
        if (pub.getEndPage() > 0) {
            // The pages.
            builder.append("pages = {");
            builder.append(pub.getStartPage());
            builder.append("--");
            builder.append(pub.getEndPage());
            builder.append("},\n");
        }
        if (pub.getDoi() != null) {
            builder.append("doi = {");
            builder.append(pub.getDoi());
            builder.append("},\n");
        }
        if (pub.getUrl() != null) {
            builder.append("url = {");
            builder.append(pub.getUrl());
            builder.append("},\n");
        }
        if (pub.getPublisher() != null) {
            // Publisher info.
            if (pub.getPublisher().getName() != null) {
                builder.append("publisher = {");
                builder.append(pub.getPublisher().getName());
                builder.append("},\n");
            }
            Address loc = pub.getPublisher().getLocation();
            if (loc != null) {
                String locString = "";
                if (loc.getCity() != null) {
                    locString += loc.getCity() + ",";
                }
                if (loc.getCountry() != null) {
                    locString += loc.getCountry();
                }
                builder.append("address = {");
                builder.append(locString);
                builder.append("},\n");
            }
        }
        if (pub.hasAuthors()) {
            // Now concatenate the authors into a valid BibTex author string.
            builder.append("author = {");
            StringBuilder authorBuilder = new StringBuilder();
            authorBuilder.append(generateAuthorBibTextEntry(pub.getAuthor(0)));
            for (int authorIndex = 1; authorIndex < pub.getNumAuthors();
                    authorIndex++) {
                authorBuilder.append(" and ");
                authorBuilder.append(generateAuthorBibTextEntry(
                        pub.getAuthor(authorIndex)));
            }
            builder.append(authorBuilder);
            builder.append("},\n");
        }
        // Final curly bracket.
        builder.append("}");
        return builder.toString();
    }

    /**
     * This method generates a BibTex-valid representation of the author name
     * and surname, to include in the author list in the BibTex entry for the
     * publication.
     *
     * @param auth Author to generate the BibTex-valid string for.
     * @return String that is the BibTex-valid representation of the author
     * info.
     */
    private static String generateAuthorBibTextEntry(Author auth) {
        if (auth == null) {
            return null;
        }
        StringBuilder authorBuilder = new StringBuilder();
        if (auth.getSurname() != null) {
            authorBuilder.append(auth.getSurname());
            if (auth.getName() != null) {
                authorBuilder.append(", ");
                String[] nameItems = auth.getName().split("\\s+");
                StringBuilder nameBuilder = new StringBuilder();
                nameBuilder.append(nameItems[0]);
                for (int i = 1; i < nameItems.length; i++) {
                    // Now abbreviate all the remaining names.
                    nameBuilder.append(" ");
                    nameBuilder.append(
                            nameItems[i].substring(0, 1).toUpperCase());
                    nameBuilder.append(".");
                }
                authorBuilder.append(nameBuilder);
            }
        } else {
            if (auth.getName() != null) {
                String[] nameItems = auth.getName().split("\\s+");
                StringBuilder nameBuilder = new StringBuilder();
                nameBuilder.append(nameItems[0]);
                for (int i = 1; i < nameItems.length; i++) {
                    // Now abbreviate all the remaining names.
                    nameBuilder.append(" ");
                    nameBuilder.append(
                            nameItems[i].substring(0, 1).toUpperCase());
                    nameBuilder.append(".");
                }
                authorBuilder.append(nameBuilder);
            }
        }
        return authorBuilder.toString();
    }

    /**
     * @param pub Publication to get the label for.
     * @return String representing a generic label that can be used for this
     * publication.
     */
    private static String getCitableLabel(Publication pub) {
        if (pub == null) {
            return "";
        }
        StringBuilder builder = new StringBuilder();
        if (pub.getTitle() != null) {
            String title = pub.getTitle();
            String[] wordItems = title.split(" ");
            for (String word : wordItems) {
                if (builder.length() + word.length() < MAX_LABEL_LENGTH
                        - RAND_LABEL_LENGTH) {
                    builder.append(word);
                }
            }
        } else if (pub.hasAuthors()) {
            for (int authorIndex = 0; authorIndex < pub.getNumAuthors();
                    authorIndex++) {
                Author auth = pub.getAuthor(authorIndex);
                String authorSurname = auth.getSurname();
                if (builder.length() + authorSurname.length()
                        < MAX_LABEL_LENGTH - RAND_LABEL_LENGTH) {
                    builder.append(authorSurname);
                }
            }
        }
        // In the end, append several random numbers to avoid possible clashes 
        // and handle degeneate cases.
        Random randa = new Random();
        for (int i = 0; i < RAND_LABEL_LENGTH; i++) {
            builder.append(randa.nextInt(10));
        }
        return builder.toString();
    }
}

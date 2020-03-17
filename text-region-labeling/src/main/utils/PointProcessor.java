package main.utils;

import javafx.util.Pair;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

public class PointProcessor {
    public static List<int[]> linkedlistToList(LinkedList<Pair<Integer, Integer>> ll) {
        List<int[]> ans = new ArrayList<>();
        Iterator<Pair<Integer, Integer>> iter = ll.descendingIterator();

        while (iter.hasNext()){
            Pair<Integer, Integer> point = iter.next();
            if (point != null) {
                ans.add(new int[]{point.getKey(), point.getValue()});
            }
        }

        return ans;
    }

    public static List<int[]> stringToList(String str) {
        String[] lst = str.split(" ");
        List<int[]> ans = new ArrayList<>(lst.length);
        for (String point : lst) {
            String[] coord = point.split(",");
            ans.add(new int[] {Integer.parseInt(coord[0]), Integer.parseInt(coord[1])});
        }

        return ans;
    }

    public static String listToString(List<int[]> lst) {
        StringBuilder sb = new StringBuilder();
        for (int[] point : lst) {
            sb.append(point[0] + "," + point[1] + " ");
        }
        return sb.toString().trim();
    }

    public static void harmonyBoundaryAndBaseLine(List<int[]> boundaryLine, List<int[]> baseLine) {
        assert boundaryLine.size() > baseLine.size();
        int startIdxBoundaryLine = boundaryLine.size() - baseLine.size();
        for (int i = 0; i < baseLine.size(); i++) {
            baseLine.get(i)[0] = boundaryLine.get(startIdxBoundaryLine + i)[0];
        }
    }
}
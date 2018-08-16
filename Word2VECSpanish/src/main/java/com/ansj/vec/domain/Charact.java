package com.ansj.vec.domain;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Charact {
	public Character ch; //具体拼音的值
	final int dim = 100;
	public Double[] C = new Double[dim]; // 音向量的值
	boolean iscp;
	public List<String> pylist = new ArrayList<String>();
	public List<Double[]> pyvec = new ArrayList<Double[]>();
	
	public Charact(Character c, boolean cp) {
		this.ch = c;
		this.iscp = cp;
		Random random = new Random();
		if(!iscp) {
			for (int j = 0; j < C.length; j++) {
				C[j] = (random.nextDouble() - 0.5) / dim;
			}
		}
	}
	
	public void setPinyin(String pinyin) {
		Random random = new Random();
		if(!pylist.contains(pinyin)) {
			pylist.add(pinyin);
			Double[] temp = new Double[dim];
			for (int j = 0; j < dim; j++) {
				temp[j] = (random.nextDouble() - 0.5) / dim;
			}
			pyvec.add(temp);
		}
	}
	
	public int getPinyinIndex(String pinyin) {
		return pylist.indexOf(pinyin);
	}
	
}

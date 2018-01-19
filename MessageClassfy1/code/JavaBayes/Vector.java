package com.cs.algorithm;

public class Vector {
    private float[] vec;
    public Vector(float[] arr){
    	vec = arr.clone();
    }
    public Vector(int n,float val){
    	vec = new float[n];
    	for(int i=0;i<n;i++){
    		vec[i] = val;
    	}
    }
    public int size(){
    	return vec.length;
    }
    public float get(int index){
    	return vec[index];
    }
    public void set(int index,float val){
    	vec[index] = val;
    }
    public Vector add(Vector v){
    	Vector newVec = new Vector(v.size(),0);
        for(int i=0;i<v.size();i++){
        	newVec.set(i, v.get(i)+this.get(i));
        }
    	return newVec;
    }
    public String toString(){
    	StringBuffer sb = new StringBuffer();
    	sb.append("[");
    	for(int i=0;i<vec.length;i++){
    		if(i == 0){
    			sb.append(vec[i]);
    		}else{
    			sb.append(",").append(vec[i]);
    		}
    	}
    	sb.append("]");
    	return sb.toString();
    }
    public void addAllOne(){
    	for(int i=0;i<vec.length;i++){
    		vec[i] += 1;
    	}
    }
    public float sum(){
    	float s = 0;
    	for(int i=0;i<vec.length;i++){
    		s += vec[i];
    	}
    	return s;
    }
    public void becomeDistribution(){
    	float s = sum();
    	for(int i=0;i<vec.length;i++){
    		vec[i] = vec[i]/s;
    	}
    }
    public void becomeMathLog(){
    	for(int i=0;i<vec.length;i++){
    		vec[i] = (float) Math.log(vec[i]);
    	}
    }
    public float multiply(Vector v){
    	float s = 0;
    	for(int i=0;i<vec.length;i++){
    		s += v.get(i)*vec[i];
    	}
    	return s;
    }
}

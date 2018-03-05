dataMat = [[1.0, -0.017612, 14.053064], [1.0, -1.395634, 4.662541], [1.0, -0.752157, 6.53862], [1.0, -1.322371, 7.152853], [1.0, 0.423363, 11.054677], [1.0, 0.406704, 7.067335], [1.0, 0.667394, 12.741452], [1.0, -2.46015, 6.866805], [1.0, 0.569411, 9.548755], [1.0, -0.026632, 10.427743], [1.0, 0.850433, 6.920334], [1.0, 1.347183, 13.1755], [1.0, 1.176813, 3.16702], [1.0, -1.781871, 9.097953], [1.0, -0.566606, 5.749003], [1.0, 0.931635, 1.589505], [1.0, -0.024205, 6.151823], [1.0, -0.036453, 2.690988], [1.0, -0.196949, 0.444165], [1.0, 1.014459, 5.754399], [1.0, 1.985298, 3.230619], [1.0, -1.693453, -0.55754], [1.0, -0.576525, 11.778922], [1.0, -0.346811, -1.67873], [1.0, -2.124484, 2.672471], [1.0, 1.217916, 9.597015], [1.0, -0.733928, 9.098687], [1.0, -3.642001, -1.618087], [1.0, 0.315985, 3.523953], [1.0, 1.416614, 9.619232], [1.0, -0.386323, 3.989286], [1.0, 0.556921, 8.294984], [1.0, 1.224863, 11.58736], [1.0, -1.347803, -2.406051], [1.0, 1.196604, 4.951851], [1.0, 0.275221, 9.543647], [1.0, 0.470575, 9.332488], [1.0, -1.889567, 9.542662], [1.0, -1.527893, 12.150579], [1.0, -1.185247, 11.309318], [1.0, -0.445678, 3.297303], [1.0, 1.042222, 6.105155], [1.0, -0.618787, 10.320986], [1.0, 1.152083, 0.548467], [1.0, 0.828534, 2.676045], [1.0, -1.237728, 10.549033], [1.0, -0.683565, -2.166125], [1.0, 0.229456, 5.921938], [1.0, -0.959885, 11.555336], [1.0, 0.492911, 10.993324], [1.0, 0.184992, 8.721488], [1.0, -0.355715, 10.325976], [1.0, -0.397822, 8.058397], [1.0, 0.824839, 13.730343], [1.0, 1.507278, 5.027866], [1.0, 0.099671, 6.835839], [1.0, -0.344008, 10.717485], [1.0, 1.785928, 7.718645], [1.0, -0.918801, 11.560217], [1.0, -0.364009, 4.7473], [1.0, -0.841722, 4.119083], [1.0, 0.490426, 1.960539], [1.0, -0.007194, 9.075792], [1.0, 0.356107, 12.447863], [1.0, 0.342578, 12.281162], [1.0, -0.810823, -1.466018], [1.0, 2.530777, 6.476801], [1.0, 1.296683, 11.607559], [1.0, 0.475487, 12.040035], [1.0, -0.783277, 11.009725], [1.0, 0.074798, 11.02365], [1.0, -1.337472, 0.468339], [1.0, -0.102781, 13.763651], [1.0, -0.147324, 2.874846], [1.0, 0.518389, 9.887035], [1.0, 1.015399, 7.571882], [1.0, -1.658086, -0.027255], [1.0, 1.319944, 2.171228], [1.0, 2.056216, 5.019981], [1.0, -0.851633, 4.375691], [1.0, -1.510047, 6.061992], [1.0, -1.076637, -3.181888], [1.0, 1.821096, 10.28399], [1.0, 3.01015, 8.401766], [1.0, -1.099458, 1.688274], [1.0, -0.834872, -1.733869], [1.0, -0.846637, 3.849075], [1.0, 1.400102, 12.628781], [1.0, 1.752842, 5.468166], [1.0, 0.078557, 0.059736], [1.0, 0.089392, -0.7153], [1.0, 1.825662, 12.693808], [1.0, 0.197445, 9.744638], [1.0, 0.126117, 0.922311], [1.0, -0.679797, 1.22053], [1.0, 0.677983, 2.556666], [1.0, 0.761349, 10.693862], [1.0, -2.168791, 0.143632], [1.0, 1.38861, 9.341997], [1.0, 0.317029, 14.739025]]
labels = [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0]
class Vector{
	constructor(arr){
		this.arr = arr;
	}
	static ones(n){
		return new Vector(this.genArr(n,1));
	}
	static genArr(n,num){
		let arr = []
		for(let i=0;i<n;i++){
			arr.push(num);
		}
		return arr;
	}
	operator(that,op){
		let new_arr = []
		for(let i=0;i<this.arr.length;i++){
			if(op == '+'){
                new_arr.push(this.arr[i]+that.arr[i])
			}
			if(op == '-'){
				new_arr.push(this.arr[i]-that.arr[i])
			}
			if(op == '*'){
                new_arr.push(this.arr[i]*that.arr[i])
			}
			if(op == '/'){
				new_arr.push(this.arr[i]/that.arr[i])
			}
			
		}
		return new_arr
	}
	multiply(that){
		let new_arr = this.operator(that,'*')
		return new Vector(new_arr)
	}
	substract(that){
		let new_arr  = this.operator(that,'-')
		return new Vector(new_arr)		
	}
	plus(that){
		let new_arr  = this.operator(that,'+')
		return new Vector(new_arr)		
	}	
	div(that){
		let new_arr  = this.operator(that,'/')
		return new Vector(new_arr)		
	}	
	toString(){
		let res = "[";
		let flag = true;
		for(let val of this.arr){
			if(flag){
				flag = false;
				res = res  +  val;
			}else{
				res = res + "," +  val;
			}
		}
		res = res + "]";
		return res
	}
	map(f){
		this.arr = this.arr.map(f)
	}
	sum(){
		let res = 0;
		for(let val of this.arr){
			res = res + val;
		}
		return res;
	}
	dot(that){
		let v = this.multiply(that)
		return v.sum()
	}
	multiplyNum(num){
		let new_arr = []
		for(let i=0;i<this.arr.length;i++){
			new_arr.push(this.arr[i]*num);
		}
		return new Vector(new_arr)
	}
}
const shape = (dataMat) =>{
	return [dataMat.length,dataMat[0].length]
}
const sigmoid = (val)=>{
	return 1.0/(1+Math.exp(-val));
}
const print = (val)=>{
	console.log(val)
}
const shuffle = (array)=>{
    var i,x,j;
    for(i=array.length;i>0;i--){
        j = Math.floor(Math.random()*i);
        x = array[j];
        array[j] = array[i-1];
        array[i-1] = x;
    }
}
const range = (n)=>{
	let arr = []
	for(let i=0;i<n;i++){
		arr.push(i)
	}
	return arr
}
const logistic = (dataMat,labels,iteratorNum=200)=>{
     const n = dataMat.length
     const m = dataMat[0].length
     let weights = Vector.ones(m)
     print(weights)
     let y = new Vector(labels)
     for(let j=0;j<iteratorNum;j++){
     	  let nums = range(n)
     	  shuffle(nums)
	      for(let i of nums){
	     	let vec = new Vector(dataMat[i])
	     	let error = labels[i] - sigmoid(vec.dot(weights))
	     	let alpha = 1/(1+i+j) + 0.01
	     	weights = weights.plus(vec.multiplyNum(error*alpha))
	      }    	
     }
     return weights
}
weights = logistic(dataMat,labels)
print(weights)
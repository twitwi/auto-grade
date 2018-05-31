
function argmin (arr) {
  let i = arr.length
  let min = Infinity
  let imin = -1
  while (i--) {
    if (arr[i] < min) {
      imin = i
      min = arr[i]
    }
  }
  return imin
}

export {argmin}

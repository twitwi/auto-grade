
import { argmin } from './tools'

function _all () {
  return { noms, prenoms, numbers, cost, bestGuess }
}

let m = new Map()
m.set('4A', 0.2)
m.set('3B', 0.2)
m.set('3g', 0.5)
m.set('3a', 0.9)
m.set('0O', 0.1)
m.set('0D', 0.11)
m.set('DO', 0.12)
m.set('zZ', 0.11)
m.set('2Z', 0.2)
m.set('2z', 0.2)
m.set('5S', 0.2)
m.set('9g', 0.2)
m.set('9G', 0.2)

m.set('_ ', 0.001)
// group of i and 1
let i1 = '1iIlL'
for (let c of i1) {
  for (let c2 of i1) {
    m.set(c + c2, 0.1)
  }
}
// capitals
for (let c of 'abcdefghijklmnopqrstuvwxyz') {
  m.set(c + c.toUpperCase(), 0.1)
}

function cost (gt, pred) {
  if (gt === pred) return 0
  let res = m.get(gt + pred)
  if (res === undefined) {
    res = m.get(pred + gt)
  }
  if (res === undefined) {
    res = 1
  }
  // console.log('cost', gt, pred, res)
  return res
}

var numbers = 'test/toto/'.split('/')
var prenoms = 'test/toto'.split('/')
var noms = 'test/toto'.split('/')

function bestGuess (group, suggestions) {
  function clean (a) { return [].filter.call(a, c => c !== '_').join('') }
  var pred = clean(group.rows.map(r => r[12]).join(''))
  function dist (sug) {
    if (sug === undefined || sug === null) return 999999
    let res = 0
    let n = pred.length
    if (pred.length !== sug.length) {
      res += 1.2345
      let nmin = Math.min(n, sug.length)
      res += 1.246 * (n - nmin)
      n = nmin
    }
    if (n === 0) {
      return res
    }
    while (--n >= 0) {
      res += cost(sug[n], pred[n])
    }
    // console.log(sug, pred, res)
    return res
  }
  let suggest = suggestions[group.name]
  if (suggest === undefined) suggest = suggestions.DEFAULT
  let dists = suggest.map(dist)
  let imax = argmin(dists)
  return [suggest[imax], imax, dists]
}

export default _all()

<template>
  <div class="ocr-questions">
    <div> <!-- to protect from vue that rebuilds the siblings on change and make a focus loss... as in https://github.com/vuejs/vue/issues/6929 -->
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    <button @click="loadBoth()">force load</button>
    <button @click="doGuess()">force do guess</button>
    <button @click="saveXlsx()">...save</button>
    <br/>

    <button @click="currentField --">«</button>
    <input v-model.number="currentField"/>
    <button @click="currentField ++">»</button>
    //
    <button @click="currentUser --">«</button>
    <input v-model.number="currentUser"/>
    <button @click="currentUser ++">»</button>
    (examid: <span v-if="orderedUsers">{{ orderedUsers[currentUser] }}</span>)
    <hr/>
    <span v-if="procfg.fields !== undefined">{{ procfg.fields.more[currentField] }}</span>
    <br/>
    {{ orderedUsers }}
    </div> <!-- see opening tag -->

    <hr/>

    »»»» <button @keydown="keydown($event)" :autofocus="'autofocus'">FOCUS</button> ««««

    <div class="for-table" v-if="currentBoxes !== undefined">
      <table class="sorting-table" ref="table">
        <thead>
          <tr>
            <th v-for="e in procfg.fields.more[currentField].propositions" :key="'th'+e"
            :style="'width:'+(100/procfg.fields.more[currentField].propositions.length)+'%'"
            :class="{ ok: procfg.fields.more[currentField].ok === e }"
            >{{ e }}</th>
          </tr>
        </thead>
        <tr v-for="(group, u) in currentBoxes" :key="group.blob + '--' + u" :class="{current: orderedUsers[currentUser] === u}">
          <td v-for="e in procfg.fields.more[currentField].propositions" :key="'td'+e">
            <span v-if="guess[currentField][u][0] === e">
              <span v-for="r in group.rows" :key="r[2]" class="scans">
                <img :src="svPath + r[13]" />
                <span class="annotation">{{r[12]}}</span>
              </span>
            </span>
          </td>
        </tr>
      </table>
    </div>

    <hr/>
    (maybe click on focus) (press 'Esc' to load and then use arrows to correct OCR guesses, use Tab and Shift-Tab to switch OCR fields)
    <br/>
    <img :src="currentFocusImage" class="focus" @click.left="currentFocusImage = ''"/>
  </div>
</template>

<script>
import config from '../customconfig'
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions
import S from '../suggestions'

export default {
  name: 'ExamId',
  data () {
    return {
      projectDir: config.defaultProjectDir,
      svPath: config.pyConnection + '/',
      procfg: {}, //   with procfg.fields.more: f ---.boxes---> [q, from=0, to=undefined]
      xlsrows: [], //   ind -> row
      currentField: 0, // f
      currentUser: 0, // indU (in this.orderedUsers)
      qBoxes: {}, //      q -> u -> boxes ([r][col])
      guess: [], //       f -> u -> guess
      suggestions: {},
      currentFocusImage: ''
    }
  },
  sockets: {
    'alert': function (data) {
      alert(data)
    },
    'got-xlsx-structured-rows': function (data) {
      // this.logs = data.map(l => ({ d: JSON.parse(l), selected: false }))
      this.xlsrows = data
    },
    'on-xlsx-saved-save-xlsx-grade': function (data) {
      this.saveXlsx(true)
    },
    'xlsx-saved': function (data) {
      alert('saved to xlsx as a new sheet')
    },
    'got-yaml-config': function (data) {
      let more = data.fields.more
      for (let k in more) {
        if (more[k].with !== undefined) {
          // TODO: consider deep assign as currently (below, with splice) we modify a shared object (that can be a feature though)
          Object.assign(more[k], data.fields.with[more[k].with])
        }
        if (more[k].boxes[0] === undefined) { // used a shortcut by just giving a q number
          more[k].boxes = [more[k].boxes]
        }
        if (more[k].ok === undefined && more[k].propositions !== undefined) { // use first proposition as ok, by default
          more[k].ok = more[k].propositions[0]
        }
        if (more[k].propositions.indexOf(more[k].ok) === -1) {
          more[k].propositions.splice(0, 0, more[k].ok)
        }
        this.guess.push({})
      }
      this.procfg = data
      // fill in suggestions
      this.suggestions = {}
      for (let k in more) {
        if (more[k].propositions !== undefined) {
          this.suggestions[more[k].name] = more[k].propositions
        }
      }
      // trigger query
      this.maybeLoadBoxForCurrentField(this.currentField)
    },
    'manual-loaded-images': function (data) {
      let q = data[Object.keys(data)[0]][0][0]
      console.log('RECEIVED BOXES FOR Q', q)
      delete data._id
      let o = {}
      Object.keys(data).map((u) => {
        let d = data[u]
        o[u] = { q, u, data: d }
      })
      this.qBoxes = Object.assign({}, this.qBoxes, { [q]: o })
      // this.$set(this.qBoxes, q, o) // not working?
      this.maybeDoGuessForCurrent()
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message']),
    currentQ () {
      if (this.procfg.fields !== undefined) {
        let desc = this.procfg.fields.more[this.currentField].boxes
        if (desc !== undefined) {
          return desc[0]
        }
      }
      return undefined
    },
    currentBoxes () {
      if (this.currentQ !== undefined) {
        let boxes = this.qBoxes[this.currentQ]
        if (boxes !== undefined) {
          return this.formatBoxesForField(boxes, this.currentField)
        }
      }
      return undefined
    },
    orderedUsers () {
      if (this.qBoxes[this.currentQ] === undefined) return undefined
      return Object.keys(this.qBoxes[this.currentQ])
    }
  },
  watch: {
    currentField (v, oldV) {
      this.maybeDoGuessForCurrent()
      this.maybeLoadBoxForCurrentField(v)
    },
    currentUser (v, oldv) {
      this.$nextTick(() => {
        this.$refs.table.querySelectorAll('.current').forEach(e => { e.scrollIntoView({ block: 'center' }) })
      })
    }
  },
  methods: {
    focus (imPath) {
      this.currentFocusImage = config.pyConnection + '/MC/' + this.projectDir + '/' + imPath
    },
    keydown (ev) {
      var k = ev.key
      var prevDef = true
      if (k === 'Escape') {
        this.loadBoth()
      } else
      if (k === 'Backspace') {
        this.currentUser = 0
      } else
      if (k === 'Enter') {
        // this.save()
      } else
      if (k === 'ArrowUp') {
        this.currentUser--
      } else
      if (k === 'ArrowDown') {
        this.currentUser++
      } else
      if (k === 'ArrowLeft') {
        let f = this.currentField
        let u = this.orderedUsers[this.currentUser]
        let conf = this.procfg.fields.more[f]
        let sug = this.suggestions[conf.name]
        let got = sug[sug.indexOf(this.guess[f][u][0]) - 1]
        if (got !== undefined) this.guess[f][u] = [got]
      } else
      if (k === 'ArrowRight') {
        let f = this.currentField
        let u = this.orderedUsers[this.currentUser]
        let sug = this.procfg.fields.more[f].propositions
        console.log(f, u, sug, this.guess[f][u], sug.indexOf(this.guess[f][u]))
        let got = sug[sug.indexOf(this.guess[f][u][0]) + 1]
        if (got !== undefined) this.guess[f][u] = [got]
      } else
      if (k === 'Tab') {
        this.currentField += ev.shiftKey ? -1 : 1
      } else
      if (k === 'x') {
        this.setCurrentToNoGuess()
      } else {
        prevDef = false
        console.log(ev)
      }
      if (prevDef) {
        ev.preventDefault()
      }
    },
    formatBoxesForField (boxes, f) {
      let conf = this.procfg.fields.more[f]
      let range = conf.boxes
      range = range.slice(1)
      let o = {}
      for (let u in boxes) {
        let d = boxes[u]
        o[u] = {
          student: u,
          blob: 'F' + f + '--',
          name: conf.name,
          rows: d.data.slice(...range)
        }
      }
      return o
    },
    maybeDoGuessForCurrent () {
      if (this.orderedUsers === undefined) return
      if (this.guess[this.currentField][this.orderedUsers[0]] !== undefined) return
      this.doGuess()
    },
    doGuess () {
      let f = this.currentField
      let boxes = this.formatBoxesForField(this.qBoxes[this.currentQ], f)
      let o = {}
      for (let u in boxes) {
        o[u] = S.bestGuess(boxes[u], this.suggestions)
        if (o[u] === null) {
          let props = this.procfg.fields.more[f].propositions
          o[u] = [props[props.lenghth - 1]]
          for (let p of this.procfg.fields.more[f].propositions) {
            if (p.replace(/±.*/, '') === '') o[u] = [p]
          }
        }
      }
      this.$set(this.guess, f, o)
    },
    setCurrentToNoGuess () {
      let f = this.currentField
      let u = this.orderedUsers[this.currentUser]
      this.$set(this.guess, f, Object.assign({}, this.guess[f], { [u]: [' '] }))
    },
    maybeLoadBoxForCurrentField (v) {
      if (this.procfg.fields === undefined) return
      let desc = this.procfg.fields.more[v].boxes
      if (desc !== undefined) {
        this.loadBoxes(desc[0])
      }
    },
    loadBoth () {
      this.loadXlsx()
      this.loadConfig()
    },
    loadXlsx () {
      this.xlsrows = []
      this.$socket.emit('xlsx-structured-rows', { pro: this.projectDir })
    },
    saveXlsx (grade = false) {
      let content = this.guess.map((m, f) => {
        let map = {}
        Object.keys(m).forEach(k => {
          map[k] = m[k][0]
          if (grade) {
            let suff = map[k].replace(/.*±/, '')
            if (suff !== map[k]) {
              map[k] = parseFloat(suff)
            } else {
              map[k] = map[k] === this.procfg.fields.more[f].ok ? 1 : 0
            }
          }
        })
        return map
      })

      let newSheet = {
        pro: this.projectDir,
        title: grade ? 'TESTOCRGrade-%d' : 'TESTOCR-%d',
        head: this.procfg.fields.more.map(f => f.name),
        content,
        callback: grade ? 'xlsx-saved' : 'on-xlsx-saved-save-xlsx-grade'
      }
      this.$socket.emit('xlsx-add-sheet', newSheet)
    },
    loadConfig () {
      this.procfg = {}
      this.$socket.emit('yaml-config', { pro: this.projectDir })
    },
    loadBoxes (q) {
      if (this.qBoxes[q] !== undefined) return
      this.qBoxes[q] = {}
      this.$socket.emit('manual-load-images', { pro: this.projectDir, '_id': 'ocrquestion', prefix: 'ocr' + q, keepImages: true, predict: true, onlyq: q, noTMP: true })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.user {
  border: 5px solid darkgrey;
}
.user span {
  font-weight: bold;
}
a {
  color: #42b983;
}
.annotated-box, .scans {
  display: inline-block;
}
.annotated-box img {
  margin-bottom: 2em;
}
.annotated-box img, .scans img {
  height: 2.5em;
}
.annotated-box .annotation {
  width: 0;
  position: relative;
  left: -1em;
}
.unguessed {
  border: 1px solid black;
  padding: 3px;
  margin: 0 2px;
}
.unguessed.current {
  border-color: red;
}
#toggle:checked ~ .offOnToggle { display: none; }

div.for-table {
  min-width: 1200px;
  height: 400px;
  margin: 0 auto;
  border: 1px solid grey;
  resize: both;
  overflow: auto;
}
div.for-table table {
  margin: 0 auto;
}
div.for-table table th {
  position: -webkit-sticky;
  position: sticky;
  top: 0;
  background: white;
}
tr.current {
  background: lightgrey;
}
th.ok {
  border: 1px solid darkgreen;
  background: darkgrey !important;
  color: black;
  font-weight: bold;
}
</style>

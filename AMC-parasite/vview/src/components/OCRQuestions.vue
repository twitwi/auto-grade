<template>
  <div class="ocr-questions">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    <button @click="loadXlsx() ; loadConfig()">load infos</button><br/>
    <button @click="iterateGuess()">do guess</button><br/>
    <button @click="saveXlsx()">...save</button><br/>

    <button @click="currentField --">«</button>
    <input v-model.number="currentField"/>
    <button @click="currentField ++">»</button>
    <br/>
    <span v-if="procfg.fields !== undefined">{{ procfg.fields.more[currentField] }}</span>

    <hr/>
    List
    <hr/>

    <div v-if="currentBoxes !== undefined">
      <div v-for="(group, u) in currentBoxes" :key="group.blob + '--' + u">
        {{ u }}
        <span v-for="r in group.rows" :key="r[2]" class="scans">
          <img :src="svPath + r[13]" />
        </span>
      </div>
    </div>

    <!--div v-for="(row,i) in xlsrows" :key="i" :class="{offOnToggle: guess[i] !== undefined}">
      <span @click="affectCurrentUnguessedToRow(i)">[ {{i}} ]</span>
      <span>{{ row[1] }}</span> --
      <span>{{ row[2] }}</span>
      ⇒
      <span v-if="guess[i]">
        {{ guess[i] }}
        <span v-for="group in boxes[guess[i]].groups" :key="group.blob">
          <span v-for="r in group.rows" :key="r[2]" class="scans">
            <img :src="svPath + r[13]" />
          </span>
        </span>
      </span>
    </div-->

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
      procfg: {}, //   with procfg.fields.more: f -> [q, from=0, to=undefined]
      xlsrows: [], //   ind -> row
      currentField: 0, // f
      qBoxes: {}, //      q -> u -> boxes ([r][col])
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
    'got-yaml-config': function (data) {
      let more = data.fields.more
      for (let k in more) {
        if (more[k].with !== undefined) {
          Object.assign(more[k], data.fields.with[more[k].with])
        }
        if (more[k].boxes[0] === undefined) { // used a shortcut by just giving a q number
          more[k].boxes = [more[k].boxes]
        }
      }
      this.procfg = data
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
          let range = this.procfg.fields.more[this.currentField].boxes
          if (this.currentQ !== range[0]) {
            console.log('WRONG', this.currentQ, range[0])
            return undefined
          }
          range = range.slice(1)
          let o = {}
          Object.keys(boxes).map((u) => {
            let d = boxes[u]
            o[u] = {
              student: u,
              blob: 'F' + this.currentField + '--',
              rows: d.data.slice(...range)
            }
          })
          return o
        }
      }
      return undefined
    }
  },
  watch: {
    currentField (v, oldV) {
      this.maybeLoadBoxForCurrentField(v)
    }
  },
  methods: {
    focus (imPath) {
      this.currentFocusImage = config.pyConnection + '/MC/' + this.projectDir + '/' + imPath
    },
    maybeLoadBoxForCurrentField (v) {
      if (this.procfg.fields === undefined) return
      let desc = this.procfg.fields.more[v].boxes
      if (desc !== undefined) {
        this.loadBoxes(desc[0])
      }
    },
    loadXlsx () {
      this.xlsrows = []
      this.$socket.emit('xlsx-structured-rows', { pro: this.projectDir })
    },
    saveXlsx () {
      let annotatedRows = this.xlsrows
      // this.xlsrows = []
      for (let ind in this.guess) {
        annotatedRows[ind][4] = parseInt(this.guess[ind])
      }
      this.$socket.emit('xlsx-structured-rows', { pro: this.projectDir, write: annotatedRows })
    },
    loadConfig () {
      this.procfg = {}
      this.$socket.emit('yaml-config', { pro: this.projectDir })
    },
    loadBoxes (q) {
      if (this.qBoxes[q] !== undefined) return
      this.qBoxes[q] = {}
      let fields = this.procfg.fields
      this.$socket.emit('manual-load-images', { pro: this.projectDir, '_id': 'ocrquestion', prefix: 'ocr' + q, keepImages: true, predict: true, onlyq: q, TMP: true })
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
</style>

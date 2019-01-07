<template>
  <div class="exam-id">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    <button @click="loadXlsx() ; loadConfig()">load parasite.xlsx</button><br/>
    <button @click="iterateGuess()">do guess</button><br/>
    <button @click="saveXlsx()">...save</button><br/>

    Unguessed :
    <span v-for="(log,u) in unguessed"
          :class="{unguessed: true, current: u===currentUnguessed}"
          :key="'ung'+u"
          @click="currentUnguessed = u"
          @click.right.prevent="focus('cr/page-'+u+'-'+boxes[u].groups[0].rows[0][4]+'.jpg')"
          >{{u}}</span>
    <hr/>
    <div v-if="boxes[currentUnguessed] !== undefined">
      {{ unguessed[currentUnguessed] }}
      <br/>
      <span v-for="group in boxes[currentUnguessed].groups" :key="'ungg'+group.blob">
        <span v-for="r in group.rows" :key="'unggg'+r[2]" class="annotated-box">
          <img :src="svPath + r[13]" />
          <span class="annotation">{{r[12]}}</span>
        </span>
      </span>
    </div>

    <hr/>
    List
    <hr/>

    <label for="toggle">
      Show only unguessed
    </label>
    <input type="checkbox" id="toggle"/>
    <div v-for="(row,i) in xlsrows" :key="i" :class="{offOnToggle: guess[i] !== undefined}">
      <span :class="{unguessed: guess[i] === undefined}" @click="affectCurrentUnguessedToRow(i)">[ {{i}} ]</span>
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
    </div>

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
      xlsrows: [], //  ind -> row
      boxes: {}, //      u -> groups (a group has .rows[r][col])
      procfg: {},
      guess: {}, //    ind -> u
      unguessed: {}, //  u -> logs
      currentUnguessed: 0, // u
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
      this.procfg = data
      this.loadBoxes()
    },
    'manual-loaded-images': function (data) {
      let get = (d, name, q, min = 0, max = undefined) => ({ name, blob: name + '--' + q + '--' + min + '--' + max, rows: d.filter(r => r[7] === q).slice(min, max) })
      delete data._id
      this.boxes = {}
      this.unguessed = {}
      Object.keys(data).map((u) => {
        let d = data[u]
        this.boxes[u] = {
          student: u,
          groups: [
            get(d, 'firstname', ...this.procfg.fields.firstname),
            get(d, 'lastname', ...this.procfg.fields.lastname)
          ]
        }
      })
      this.guess = {}
      for (let i in this.xlsrows) {
        if (this.xlsrows[i][4] !== null) {
          this.guess[i] = '' + this.xlsrows[i][4] // we use string as "k in this.boxes" seems to be string
        }
      }
      // this.iterateGuess()
      // this.fillGuesses()
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message'])
  },
  methods: {
    iterateGuess () {
      this.unguessed = {}
      this.fillGuesses([...Object.values(this.guess)], (o, i) => this.guess[i] !== undefined ? null : o)
    },
    fillGuesses (skip = [], filt = o => o) {
      let upper = v => v === null ? null : v.toUpperCase()
      let suggestions = {
        firstname: this.xlsrows.map(o => upper(o[1])).map(filt),
        lastname: this.xlsrows.map(o => upper(o[2])).map(filt)
      }
      console.log(suggestions)
      for (let k in this.boxes) {
        if (skip.indexOf(k) !== -1) continue
        let guessFirstname = S.bestGuess(this.boxes[k].groups[0], suggestions)
        let guessLastname = S.bestGuess(this.boxes[k].groups[1], suggestions)
        let ok = false
        if (guessFirstname !== null && guessLastname !== null) {
          if (guessFirstname[1] === guessLastname[1] || suggestions.firstname[guessLastname[1]] === guessFirstname[0]) {
            if (this.guess[guessLastname[1]] === undefined || this.guess[guessLastname[1]] === k) {
              this.guess[guessLastname[1]] = k
              ok = true
            }
          }
        }
        if (!ok) {
          if (guessFirstname === null) guessFirstname = [-1, -2, -3]
          if (guessLastname === null) guessLastname = [-1, -2, -3]
          console.log(k, ':', guessFirstname[1] === guessLastname[1], guessFirstname[1], guessLastname[1], guessFirstname[0], guessLastname[0])
          this.unguessed[k] = { k, log: [k, ':', guessFirstname[1] === guessLastname[1], guessFirstname[1], guessLastname[1], guessFirstname[0], guessLastname[0]] }
        }
      }
    },
    affectCurrentUnguessedToRow (i) {
      if (this.unguessed[this.currentUnguessed] === undefined) return
      this.$delete(this.unguessed, this.currentUnguessed)
      this.$set(this.guess, i, this.currentUnguessed)
      this.currentUnguessed = 0
    },
    focus (imPath) {
      this.currentFocusImage = config.pyConnection + '/MC/' + this.projectDir + '/' + imPath
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
    loadBoxes () {
      this.boxes = {}
      let fields = this.procfg.fields
      if (fields.lastname[0] !== fields.firstname[0]) {
        alert('Unimplemented: firstname and lastname in different questions')
        return
      }
      this.$socket.emit('manual-load-images', { pro: this.projectDir, '_id': 'examid', predict: true, onlyq: fields.lastname[0], noTMP: true })
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

//Variables//
const rect = document.querySelector(".rect");
const modal = document.querySelector(".modal");
const overlay = document.querySelector(".overlay");
const btnCloseModal = document.querySelector(".close-modal");
const loader = document.querySelector(".loader-wrap");
const select = document.querySelector("#select");
const htmlBtn = `<div class="btn pre">Our Prediction</div>`;
const countrySet = new Set();
let today = new Date();
let hasHappened = false;
let currTimeStamp;
let currDescr;
let todaysEvent;
let currEvent;

//Side functions//

//Using this to get the date out of a timestamp
const dateFormat = (date) => {
  const currDate = new Date(date * 1000);
  return currDate;
};

//Returns a good looking time format - if 3 hours and 5 minutes -> 03:05, if 13 hours and 15 minutes -> 13:15
const timeFormat = (time) => {
  if (time / 10 < 1) return `0${time}`;
  return time;
};

//Makes the modal visible and sticky
const openModal = function () {
  modal.classList.remove("hidden");
  overlay.classList.remove("hidden");
  modal.style.position = "fixed";
  modal.style.zIndex = "100";
  overlay.style.position = "fixed";
};

//Makes the modal unvisible
const closeModal = function () {
  modal.classList.add("hidden");
  overlay.classList.add("hidden");
};

//Requesting data out of the API
const options = {
  method: "GET",
  headers: {
    "X-RapidAPI-Key": "fa75f7cffbmsh2f16132e68bbfdap19ebe4jsnb8ea9113474e",
    "X-RapidAPI-Host": "footapi7.p.rapidapi.com",
  },
};
console.log("test!!!!!");
//Using today's date to get today's matches
fetch(
  `https://footapi7.p.rapidapi.com/api/matches/${today.getDate()}/${
    today.getMonth() + 1
  }/${today.getFullYear()}`,
  options
)
  .then((response) => response.json())
  .then((response) => {
    todaysEvent = response.events;
    console.log(response);
    for (let i = 0; i < response.events.length; i++) {
      currEvent = todaysEvent[i];
      currTimeStamp = currEvent.startTimestamp;
      currDescr = currEvent.status.description;

      if (currEvent.tournament.category.name.includes("Amateur")) {
        continue;
      }

      if (
        currDescr != "Not started" &&
        currDescr != "Postponed" &&
        currDescr != "Canceled"
      ) {
        hasHappened = true;
      }

      const htmlClubs = `<p class="clubs c${i}">${currEvent.awayTeam.name} vs ${currEvent.homeTeam.name}
      (${currDescr})</p>`;
      rect.insertAdjacentHTML("beforeend", htmlClubs);

      const htmlScore = `<p class= "score s${i}">${
        hasHappened
          ? currEvent.awayScore.display
          : timeFormat(dateFormat(currTimeStamp).getHours())
      } ${hasHappened ? "-" : ":"} ${
        hasHappened
          ? currEvent.homeScore.display
          : timeFormat(dateFormat(currTimeStamp).getMinutes())
      }</p>`;

      const htmlDate = `<p class= "dateVis">${dateFormat(
        currTimeStamp
      ).getDate()}/${dateFormat(currTimeStamp).getMonth() + 1}</p>`;

      document
        .querySelector(`.c${i}`)
        .insertAdjacentHTML("beforeend", htmlScore);
      document.querySelector(`.c${i}`).insertAdjacentHTML("beforeend", htmlBtn);
      document
        .querySelector(`.c${i}`)
        .insertAdjacentHTML("beforeend", htmlDate);

      if (hasHappened)
        document.querySelector(`.s${i}`).style.background = "#088F8F";
      else document.querySelector(`.s${i}`).style.background = "#C0C0C0";

      hasHappened = false;
    }
  })
  .catch((err) => console.error(err))
  .finally(() => {
    for (let i = 0; i < document.querySelectorAll(".pre").length; i++)
      document.querySelectorAll(".pre")[i].addEventListener("click", openModal);

    btnCloseModal.addEventListener("click", closeModal);

    for (let i = 0; i < todaysEvent.length; i++) {
      if (!todaysEvent[i].tournament.category.name.includes("Amateur"))
        countrySet.add(todaysEvent[i].tournament.category.name);
    }
    countrySet.forEach((country) => {
      const countryHTML = `<option>${country}</option>`;
      select.insertAdjacentHTML("beforeend", countryHTML);
    });
    select.addEventListener("change", () => {
      for (let i = 0; i < todaysEvent.length; i++) {
        if (todaysEvent[i].tournament.category.name.includes("Amateur"))
          continue;
        if (select.value == "All") {
          document.querySelector(`.c${i}`).classList.remove("hidden");
        } else if (select.value != todaysEvent[i].tournament.category.name) {
          document.querySelector(`.c${i}`).classList.add("hidden");
        } else {
          document.querySelector(`.c${i}`).classList.remove("hidden");
        }
      }
    });
  })
  .then(() => (loader.style.display = "none"));
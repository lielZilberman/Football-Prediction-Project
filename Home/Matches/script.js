const rect = document.querySelector(".rect");
const modal = document.querySelector(".modal");
const overlay = document.querySelector(".overlay");
const btnCloseModal = document.querySelector(".close-modal");
const htmlBtn = `<div class="btn">Our Prediction</div>`;
let today = new Date();
let hasHappened = false;
let currTimeStamp;
let currDescr;
let currEvent;

const dateFormat = (date) => {
  const currDate = new Date(date * 1000);
  return currDate;
};

const timeFormat = (time) => {
  if (time / 10 < 1) return `0${time}`;
  return time;
};

const scrollModal = () => {
  modal.style.top = `${window.scrollY + 270}px`;
  overlay.style.top = `${window.scrollY}px`;
};

const openModal = function () {
  // document.body.style.position = "fixed";
  modal.style.top = `${window.scrollY + 270}px`;
  overlay.style.top = `${window.scrollY}px`;
  modal.classList.remove("hidden");
  overlay.classList.remove("hidden");
  document.body.classList.add(".stop-scrolling");
  window.addEventListener("scroll", scrollModal);
};

const closeModal = function () {
  modal.classList.add("hidden");
  overlay.classList.add("hidden");
};

const options = {
  method: "GET",
  headers: {
    "X-RapidAPI-Key": "fa75f7cffbmsh2f16132e68bbfdap19ebe4jsnb8ea9113474e",
    "X-RapidAPI-Host": "footapi7.p.rapidapi.com",
  },
};
fetch(
  `https://footapi7.p.rapidapi.com/api/matches/top/${today.getDate()}/${
    today.getMonth() + 1
  }/${today.getFullYear()}`,
  options
)
  .then((response) => response.json())
  .then((response) => {
    console.log(response);
    for (let i = 0; i < response.events.length; i++) {
      currEvent = response.events[i];
      currTimeStamp = currEvent.startTimestamp;
      currDescr = currEvent.status.description;

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
    for (let i = 0; i < document.querySelectorAll(".btn").length; i++)
      document.querySelectorAll(".btn")[i].addEventListener("click", openModal);

    btnCloseModal.addEventListener("click", closeModal);
  });
// fetch("https://footapi7.p.rapidapi.com/api/team/2672/image", options)
//   .then((response) => response.json())
//   .then((response) => console.log(response))
//   .catch((err) => console.error(err));

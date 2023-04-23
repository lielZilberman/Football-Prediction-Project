const rect = document.querySelector(".rect");
const modal = document.querySelector(".modal");
const overlay = document.querySelector(".overlay");
const btnCloseModal = document.querySelector(".close-modal");
const htmlBtn = `<div class="btn">Our Prediction</div>`;
let today = new Date();
let isStarted = false;

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
    for (let i = 0; i < 200; i++) {
      if (
        response.events[i].status.description != "Not started" &&
        response.events[i].status.description != "Postponed"
      ) {
        isStarted = true;
      }
      const htmlClubs = `<p class="clubs c${i}">${response.events[i].awayTeam.name} vs ${response.events[i].homeTeam.name}
      (${response.events[i].status.description})</p>`;
      rect.insertAdjacentHTML("beforeend", htmlClubs);
      const htmlScore = `<p class= "score">${
        isStarted
          ? response.events[i].awayScore.display
          : timeFormat(dateFormat(response.events[i].startTimestamp).getHours())
      } ${isStarted ? "-" : ":"} ${
        isStarted
          ? response.events[i].homeScore.display
          : timeFormat(
              dateFormat(response.events[i].startTimestamp).getMinutes()
            )
      }</p>`;
      document
        .querySelector(`.c${i}`)
        .insertAdjacentHTML("beforeend", htmlScore);
      document.querySelector(`.c${i}`).insertAdjacentHTML("beforeend", htmlBtn);
      isStarted = false;
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
